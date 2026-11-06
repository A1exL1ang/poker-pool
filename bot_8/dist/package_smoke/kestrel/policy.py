"""Position-aware tournament policy with bounded, opponent-weighted equity."""

from dataclasses import dataclass
from time import perf_counter_ns

from .cards import EMPTY, extend, score, hand_class, draw_strength
from .equity import BASE, PERCENTILE, preflop_equity, pool_index, estimate, made_quality
from .icm import call_threshold
from .model import Action, Observation, legalize, contestable_pots


@dataclass(slots=True)
class Opponent:
    opportunities: int = 0
    voluntary: int = 0
    raises: int = 0
    faced: int = 0
    folds: int = 0
    post_actions: int = 0
    post_raises: int = 0
    post_faced: int = 0
    post_folds: int = 0

    @property
    def vpip(self):
        return (6.4 + self.voluntary) / (20 + self.opportunities)

    @property
    def pfr(self):
        return (4.4 + self.raises) / (20 + self.opportunities)

    @property
    def fold_rate(self):
        return (7.5 + self.post_folds) / (15 + self.post_faced)

    @property
    def pre_fold_rate(self):
        return (7.5 + self.folds - self.post_folds) / (15 + self.faced - self.post_faced)

    @property
    def aggression(self):
        return (5.5 + self.post_raises) / (20 + self.post_actions)


def _mix(seed):
    seed ^= seed >> 16
    seed = seed * 0x7FEB352D & 0xFFFFFFFF
    seed ^= seed >> 15
    seed = seed * 0x846CA68B & 0xFFFFFFFF
    return (seed ^ (seed >> 16)) & 0xFFFFFFFF


def _seed(obs):
    x = (obs.hand_id * 0x9E3779B1 + obs.hero * 0x45D9F3B + obs.pot * 71 + obs.to_call * 31) & 0xFFFFFFFF
    for c in obs.hole + obs.board:
        x = _mix(x ^ (c + 1))
    return x


def _pre_order(obs):
    seats = sorted(p.seat for p in obs.players)
    start = seats.index(obs.button)
    if len(seats) == 2:
        return seats[start:] + seats[:start]
    start = (start + 3) % len(seats)
    return seats[start:] + seats[:start]


def _in_position(obs):
    live = sorted(p.seat for p in obs.players if not p.folded)
    after = [s for s in live if s > obs.button] + [s for s in live if s <= obs.button]
    return bool(after and after[-1] == obs.hero)


def _open_width(behind):
    return (1.0, .57, .45, .31, .25, .21, .175, .15)[min(7, behind)]


class Agent:
    """One instance per table; ``act(Observation|dict) -> Action``.

    Table identity changes and explicit reset() clear all opponent state.
    Deterministic samples, with an optional wall-clock work budget. Disable the
    budget for exactly reproducible offline studies; it is enabled by default.
    ``diagnostics`` is local inspection only; the gameplay action is tiny.
    """

    def __init__(self, sample_scale=.5, time_budget_us=450):
        self.sample_scale = sample_scale
        self.time_budget_ns = None if time_budget_us is None else int(time_budget_us * 1000)
        self.reset()

    def reset(self, table_id=None):
        self.table_id = table_id
        self.hand_id = None
        self.seen_event = -1
        self.stats = {}
        self.pre_seen = set()
        self.pre_voluntary = set()
        self.pre_raised = set()
        self.diagnostics = {}

    def _observe(self, obs):
        if self.table_id != obs.table_id:
            self.reset(obs.table_id)
        if self.hand_id != obs.hand_id:
            self.hand_id = obs.hand_id
            self.seen_event = -1
            self.pre_seen.clear()
            self.pre_voluntary.clear()
            self.pre_raised.clear()
        for event in obs.history:
            if event.id <= self.seen_event:
                continue
            self.seen_event = event.id
            if event.seat == obs.hero or event.action not in ("fold", "check", "call", "raise"):
                continue
            st = self.stats.setdefault(event.seat, Opponent())
            if event.street == "preflop":
                if event.seat not in self.pre_seen:
                    self.pre_seen.add(event.seat)
                    st.opportunities += 1
                if event.action in ("call", "raise") and event.seat not in self.pre_voluntary:
                    self.pre_voluntary.add(event.seat)
                    st.voluntary += 1
                if event.action == "raise" and event.seat not in self.pre_raised:
                    self.pre_raised.add(event.seat)
                    st.raises += 1
            else:
                st.post_actions += 1
                st.post_raises += event.action == "raise"
            if event.to_call > 0:
                st.faced += 1
                st.folds += event.action == "fold"
                if event.street != "preflop":
                    st.post_faced += 1
                    st.post_folds += event.action == "fold"

    def _ranges(self, obs, villains):
        order = _pre_order(obs)
        widths = {p.seat: 1.0 for p in villains}
        aggression = {p.seat: 0.0 for p in villains}
        passive = {p.seat: 0.0 for p in villains}
        raises = 0
        for event in obs.history:
            seat = event.seat
            if event.street == "preflop":
                if event.action == "raise":
                    raises += 1
                if seat not in widths:
                    continue
                st = self.stats.get(seat, Opponent())
                if event.action == "raise":
                    if raises == 1:
                        behind = len(order) - 1 - order.index(seat)
                        width = _open_width(behind) * (.65 + 1.6 * st.pfr)
                        # Short stack open jams can be much wider than deep opens.
                        if event.amount <= 12 * obs.big_blind and event.amount >= 5 * obs.big_blind:
                            width *= 1.15
                    else:
                        width = (.09 if raises == 2 else .035) * (.65 + 1.6 * st.pfr)
                    widths[seat] = min(widths[seat], max(.03, min(.85, width)))
                elif event.action == "call":
                    width = (min(.85, st.vpip * 1.7) if raises == 0 else
                             min(.55, st.vpip * .95) if raises == 1 else .12 if raises == 2 else .055)
                    if event.to_call > 10 * obs.big_blind:
                        width *= .72
                    widths[seat] = min(widths[seat], max(.03, width))
            elif seat in widths:
                fade = 1.0 if event.street == obs.street else .35
                if event.action == "raise":
                    fraction = event.amount / max(obs.big_blind, event.pot_before)
                    aggression[seat] += fade * (1.0 + min(1.5, fraction) + (1.0 if event.to_call else 0))
                elif event.action == "call":
                    passive[seat] += .65 * fade
        profiles = []
        for p in villains:
            st = self.stats.get(p.seat, Opponent())
            bluff = min(.25, max(.04, .07 + .34 * (st.aggression - .20)))
            profiles.append((pool_index(widths[p.seat]), min(4.5, aggression[p.seat]),
                             min(1.8, passive[p.seat]), bluff))
        return widths, profiles

    def _risk(self, obs, h, villains, call):
        odds = call / max(1, obs.pot + call)
        if not obs.payouts or obs.payouts[0] == obs.payouts[-1]:
            return 0.0
        # A winner-take-all schedule is exactly chip EV under ICM.
        if all(x == obs.payouts[-1] for x in obs.payouts[1:]):
            return 0.0
        if (len(villains) == 1 and call >= .25 * h.stack
                and h.invested + call >= max(p.invested for p in obs.players)):
            return max(0.0, min(.35, call_threshold(obs, call, villains[0].seat) - odds))
        scale = obs.payouts[0] - obs.payouts[-1]
        flat = sum(x - obs.payouts[-1] for x in obs.payouts[1:]) / max(1e-12, scale * (len(obs.players) - 1))
        covered = any(p.stack >= h.stack for p in villains)
        fraction = min(1.0, call / max(1, h.stack))
        return min(.15, (.018 + .15 * flat) * fraction ** .6 * (1.25 if covered else .7))

    def act(self, observation):
        deadline = None if self.time_budget_ns is None else perf_counter_ns() + self.time_budget_ns
        obs = Observation.from_dict(observation) if isinstance(observation, dict) else observation
        self._observe(obs)
        h = next(p for p in obs.players if p.seat == obs.hero)
        villains = [p for p in obs.players if p.seat != obs.hero and not p.folded]
        if not villains:
            return Action("check" if not obs.to_call else "call")
        seed = _seed(obs)
        widths, profiles = self._ranges(obs, villains)
        if not obs.board:
            action = self._preflop(obs, h, villains, widths, seed)
        else:
            action = self._postflop(obs, h, villains, profiles, seed, deadline)
        return legalize(obs, action)

    def _preflop(self, obs, h, villains, widths, seed):
        bb = obs.big_blind
        hand = hand_class(*obs.hole)
        percentile = PERCENTILE[hand]
        a, b = (c >> 2 for c in obs.hole)
        suited = (obs.hole[0] & 3) == (obs.hole[1] & 3)
        pair = a == b
        order = _pre_order(obs)
        behind = len(order) - order.index(obs.hero) - 1
        ip = _in_position(obs)
        raises = [e for e in obs.history if e.street == "preflop" and e.action == "raise"]
        limpers = sum(e.action == "call" and e.street == "preflop" and e.to_call <= bb for e in obs.history)
        effective = min(h.stack + h.committed, max(p.stack + p.committed for p in villains))
        depth = effective / bb
        call = min(obs.to_call, h.stack)
        eligible_pot, _ = contestable_pots(obs, call, villains)
        risk = self._risk(obs, h, villains, call)
        roll = (seed & 0xFFFF) / 65536
        adjusted = percentile - (.035 if suited else 0) - (.018 if 1 <= abs(a-b) <= 2 and min(a,b) >= 3 else 0)
        self.diagnostics = {"street": "preflop", "class": hand, "percentile": round(percentile, 4), "depth_bb": round(depth, 2)}
        if not raises:
            opening = _open_width(behind)
            if len(obs.players) == 2:
                opening = .80 if obs.hero == obs.button else 1.0
            opening -= min(.10, limpers * .035)
            if depth <= 12 and (obs.to_call <= bb or limpers):
                # Chip-EV jam approximation with at least one caller. Conservative
                # multiway correction avoids treating seven possible callers as one.
                n = max(1, behind + limpers)
                call_range = min(.35, .10 + .75 / max(depth, 2))
                fold = (1 - call_range) ** n
                q = preflop_equity(hand, call_range) * (1 - min(.10, .018 * (n - 1)))
                stake = max(1, effective - h.committed)
                extra = max(0, effective - bb)
                ev = fold * obs.pot + (1-fold) * (q * (obs.pot + extra + stake) - stake)
                jam_tax = self._risk(obs, h, villains, min(h.stack, stake)) * stake
                if ev > .08 * bb + jam_tax and adjusted < min(.90, opening + .12):
                    return Action("raise", h.stack + h.committed)
                if obs.to_call == 0:
                    return Action("check")
                return Action("fold")
            if adjusted <= opening and (not limpers or adjusted < .27 + (.08 if ip else 0)):
                size = (2.15 if depth >= 25 else 2.0) + .9 * limpers
                if behind == 1 and len(obs.players) > 2:
                    size = max(size, 2.65)
                return Action("raise", round(size * bb))
            if not obs.to_call:
                return Action("check")
            if limpers and call <= .04 * effective and (pair or (suited and abs(a-b) <= 3 and min(a,b) >= 2)):
                return Action("call")
            # Small blind completion gets a price but realizes less equity OOP.
            if call <= .5 * bb and limpers and BASE[hand] > .47 and call / max(1, obs.pot + call) < .19:
                return Action("call")
            return Action("fold")
        raiser = raises[-1].seat
        width = widths.get(raiser, .12)
        q = preflop_equity(hand, width)
        callers = sum(1 for p in villains if p.committed >= h.committed + obs.to_call and p.seat != raiser)
        q *= (.87 ** callers)
        odds = call / max(1, eligible_pot + call)
        terminal = call >= h.stack or all(p.all_in for p in villains)
        realization = 1.0 if terminal else (.94 if ip else .84)
        if not terminal and (suited or pair):
            realization += .04
        strength = q * min(1.0, realization)
        self.diagnostics.update(equity=round(q, 4), odds=round(odds, 4), risk=round(risk, 4), range=round(width, 3))
        if not terminal and obs.can_raise:
            current = h.committed + obs.to_call
            value = (q >= .60 and percentile <= .085) or (a == b and a >= 10)
            if len(raises) >= 2:
                value = (pair and a >= 11) or (q >= .56 and percentile < .035)
            if value:
                target = round(current * (3.1 if ip else 3.8))
                if depth <= 20 or target >= .37 * effective:
                    target = h.stack + h.committed
                return Action("raise", target)
            st = self.stats.get(raiser, Opponent())
            blocker = (a == 12 or b == 12) and suited
            if (len(raises) == 1 and depth >= 30 and ip and blocker and width > .27
                    and st.pre_fold_rate > .53 and roll < .11 and callers == 0):
                return Action("raise", round(current * 3.1))
        if strength >= odds + risk + (.012 if terminal else .035):
            # Defend at an attractive price, but avoid stacking dominated hands
            # against very tight, deep reraises just because pot odds look small.
            if len(raises) >= 2 and not terminal and percentile > .24 and call > .08 * effective:
                return Action("fold")
            return Action("call")
        if pair and not terminal and call <= .045 * effective and len(raises) <= 2:
            return Action("call")
        return Action("fold" if obs.to_call else "check")

    def _postflop(self, obs, h, villains, profiles, seed, deadline):
        call = min(obs.to_call, h.stack)
        eligible_pot, pots = contestable_pots(obs, call, villains)
        ip = _in_position(obs)
        n = len(villains)
        value = score(extend(extend(EMPTY, obs.board), obs.hole))
        quality = made_quality(obs.hole, obs.board, value)
        draw, nut_blocker = draw_strength(obs.hole, obs.board)
        effective = min(h.stack, max(p.stack for p in villains))
        spr = effective / max(1, obs.pot + call)
        terminal = len(obs.board) == 5 or call >= h.stack or all(p.all_in for p in villains)
        risk = self._risk(obs, h, villains, call)
        q, ess, error = estimate(obs.hole, obs.board, profiles, seed, self.sample_scale, deadline, pots)
        odds = call / max(1, eligible_pot + call)
        uncertainty = min(.045, .22 * error)
        realized = q - uncertainty
        if not terminal:
            realization = .95 if ip else .87
            realization -= .025 * (n - 1)
            if quality > .82 or draw > .3:
                realization += .04
            realized = q * min(1., realization) - uncertainty
        self.diagnostics = {"street": obs.street, "equity": round(q, 4), "ess": round(ess, 1),
                            "stderr": round(error, 4), "odds": round(odds, 4), "risk": round(risk, 4)}
        roll = (seed & 0xFFFF) / 65536
        raisable = obs.can_raise and any(p.stack > 0 for p in villains)
        if call:
            if raisable and q > (.81 if n == 1 else .76) + risk and quality >= .83:
                target = h.committed + call + round(.72 * (obs.pot + call))
                if target - h.committed >= .58 * h.stack or spr < 1.2:
                    target = h.stack + h.committed
                return Action("raise", target)
            if (raisable and n == 1 and len(obs.board) < 5 and draw > .38 and nut_blocker
                    and spr >= 2 and call < .14 * h.stack and roll < .12 and q > .32):
                return Action("raise", h.committed + call + round(.80 * (obs.pot + call)))
            implied = .018 if draw > .25 and ip and call < .10 * h.stack else 0.0
            if realized + implied >= odds + risk + .008:
                return Action("call")
            return Action("fold")
        if not raisable:
            return Action("check")
        pot = max(obs.big_blind, obs.pot)
        if q > (.70 if n == 1 else .61) and quality > .63:
            fraction = .72 if q > .86 else .52
            if len(obs.board) == 5 and q > .92:
                fraction = .9
            target = h.committed + max(obs.big_blind, round(pot * fraction))
            if effective <= pot * 1.25 and q > .77 and quality > .75:
                target = h.stack + h.committed
            return Action("raise", target)
        if n == 1 and q > .59 and quality > .70 and (ip or len(obs.board) < 5):
            return Action("raise", h.committed + max(obs.big_blind, round(.36 * pot)))
        aggressors = [e.seat for e in obs.history if e.action == "raise"]
        initiative = bool(aggressors and aggressors[-1] == obs.hero)
        board_ranks = [c >> 2 for c in obs.board]
        suit_counts = [sum((c & 3) == s for c in obs.board) for s in range(4)]
        dry = max(suit_counts) <= 2 and (len(set(board_ranks)) < len(board_ranks) or max(board_ranks) - min(board_ranks) >= 7)
        fold_rate = sum(self.stats.get(p.seat, Opponent()).fold_rate for p in villains) / n
        if len(obs.board) == 3 and n <= 2:
            freq = (.60 if n == 1 else .20) * (1.12 if dry else .7)
            freq *= .75 + fold_rate
            if initiative and roll < freq and (q > .25 or draw > .15) and (ip or dry):
                return Action("raise", h.committed + max(obs.big_blind, round(.33 * pot)))
        if n == 1 and len(obs.board) < 5 and draw > .25 and roll < (.58 if ip else .35):
            return Action("raise", h.committed + max(obs.big_blind, round(.60 * pot)))
        river_blocker = any((c >> 2) == 12 and suit_counts[c & 3] >= 3 for c in obs.hole)
        bluff_frequency = (.22 if river_blocker else .10) if len(obs.board) == 5 else .12
        if n == 1 and ip and q < .32 and roll < bluff_frequency and fold_rate >= .48:
            # Infrequent polarized stab with poor showdown value. On the river,
            # an ace of the board's flush suit increases the bluff frequency.
            return Action("raise", h.committed + max(obs.big_blind, round(.66 * pot)))
        return Action("check")
