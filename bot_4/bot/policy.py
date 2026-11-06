"""Independent position/range-aware tournament policy with bounded inference."""
from dataclasses import dataclass
from .cards import preflop_index, features
from .interface import State, Action, legalize
from .model import PREFLOP, predict
from .icm import risk_premium


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


@dataclass(slots=True)
class Tendencies:
    hands: int = 0
    vpip: int = 0
    raises: int = 0
    bets: int = 0
    passive: int = 0
    faced: int = 0
    folds: int = 0

    @property
    def looseness(self):
        return (self.vpip + 9.) / (self.hands + 30.)

    @property
    def pfr(self):
        return (self.raises + 5.4) / (self.hands + 30.)

    @property
    def aggression(self):
        return (self.bets + 4.) / (self.bets + self.passive + 12.)

    @property
    def fold_rate(self):
        return (self.folds + 6.) / (self.faced + 15.)


class Agent:
    """Construct once per table; act(dict) or decide(State) for each decision.

    Public history is consumed incrementally. reset() discards all learned
    tendencies. No network, subprocess, clock reads or live rollout searches.
    """
    def __init__(self, seed=404):
        self.seed = seed
        self.reset()

    def reset(self, table_id=None):
        self.table_id = table_id
        self.stats = {}
        self.hand_id = None
        self.seen = 0
        self.vpipped = set()
        self.raised = set()
        self.cache_key = None
        self.cache = None
        self.rng = self.seed or 1

    def _random(self):
        x = self.rng
        x ^= (x << 13) & 0xffffffff
        x ^= x >> 17
        x ^= (x << 5) & 0xffffffff
        self.rng = x & 0xffffffff
        return self.rng / 4294967296.

    def _observe(self, state):
        if self.table_id != state.table_id:
            self.reset(state.table_id)
        if self.hand_id != state.hand_id:
            self.hand_id = state.hand_id
            self.seen = 0
            self.vpipped.clear()
            self.raised.clear()
            for p in state.players:
                if p.stack + p.contributed:
                    self.stats.setdefault(p.seat, Tendencies()).hands += 1
        for e in state.history[self.seen:]:
            t = self.stats.setdefault(e.seat, Tendencies())
            if e.street == "preflop":
                if e.action in ("call", "raise") and e.seat not in self.vpipped:
                    t.vpip += 1
                    self.vpipped.add(e.seat)
                if e.action == "raise" and e.seat not in self.raised:
                    t.raises += 1
                    self.raised.add(e.seat)
            else:
                if e.action == "raise":
                    t.bets += 1
                elif e.action in ("check", "call"):
                    t.passive += 1
                if e.to_call > 0:
                    t.faced += 1
                    t.folds += e.action == "fold"
        self.seen = len(state.history)

    def observe(self, state):
        """Optional callback, including final public history after folding."""
        self._observe(State.from_dict(state) if isinstance(state, dict) else state)

    def act(self, observation):
        return self.decide(State.from_dict(observation)).to_dict()

    @staticmethod
    def _order(s, preflop=False):
        seats = sorted(p.seat for p in s.players if p.stack + p.contributed > 0)
        i = seats.index(s.button)
        offset = (0 if len(seats) == 2 else 3) if preflop else 1
        i = (i + offset) % len(seats)
        return seats[i:] + seats[:i]

    @staticmethod
    def _range_equity(ix, width):
        es = PREFLOP["versus"][ix]
        if width >= .5:
            return es[1] + (es[0] - es[1]) * min(1., (width - .5) * 2)
        if width >= .2:
            return es[2] + (es[1] - es[2]) * ((width - .2) / .3)
        return es[3] + (es[2] - es[3]) * clamp((width - .05) / .15, 0., 1.)

    def decide(self, state):
        self._observe(state)
        hero = next(p for p in state.players if p.seat == state.hero)
        if hero.stack <= 0 or hero.folded:
            return Action("check")
        opponents = [p for p in state.players if p.seat != state.hero and not p.folded and p.stack + p.contributed > 0]
        if not opponents:
            return Action("check")
        if not state.board:
            choice = self._preflop(state, hero, opponents)
        else:
            choice = self._postflop(state, hero, opponents)
        return legalize(state, choice)

    def _preflop(self, s, h, opponents):
        bb = s.big_blind
        call = min(s.to_call, h.stack)
        eligible_cap = h.contributed + call
        pot = max(1, s.pot - sum(max(0, p.contributed - eligible_cap) for p in s.players)) if call == h.stack else s.pot
        order = self._order(s, True)
        hi = order.index(s.hero)
        behind = [p for p in opponents if p.stack > 0 and order.index(p.seat) > hi]
        raises = [e for e in s.history if e.street == "preflop" and e.action == "raise"]
        limpers = sum(p.committed >= bb and p.seat not in order[-2:] for p in opponents)
        eff = max(0, min(h.stack, max(p.stack + p.committed - h.committed for p in opponents)))
        depth = (eff + h.committed) / bb
        ix = preflop_index(s.hole)
        percentile = PREFLOP["percentile"][ix]
        a, b = s.hole
        r1, r2 = max(a >> 2, b >> 2), min(a >> 2, b >> 2)
        suited = (a & 3) == (b & 3)
        play = (.035 if suited else 0.) + (.02 if suited and r1 - r2 <= 2 else 0.)
        effective_percentile = max(0., percentile - play * min(1., depth / 35))
        active = [p for p in s.players if p.stack + p.contributed > 0]
        risk = risk_premium(active, s.hero, s.payouts, min(1., call / max(h.stack, 1)))

        if not raises:
            nbehind = len(behind)
            opening = (.82 if len(active) == 2 else (.68, .58, .43, .31, .24, .20, .17, .15)[min(7, nbehind)])
            if len(active) == 2 and s.hero != s.button and any(p.committed >= bb for p in opponents):
                opening = .50
            opening += .03 * min(3., s.ante * len(active) / bb)
            if limpers:
                opening = min(opening, .30 - .025 * min(limpers - 1, 3))
                if s.hero in order[-2:]:
                    opening = min(opening, .23)
            # Jam below 11 BB. The EV includes fold equity, called equity,
            # risk premium, and existing chips; no invented push/fold chart.
            if depth <= 11 and s.can_raise and s.max_raise_to > h.committed + call:
                candidates = behind if not limpers else opponents
                foldprob = 1.
                widths = []
                for p in candidates:
                    t = self.stats[p.seat]
                    width = clamp(.10 + .018 * (11 - depth) + .35 * (t.looseness - .3), .07, .52)
                    if p.committed >= bb and p.seat not in order[-2:]:
                        width = max(width, .35)
                    widths.append(width)
                    foldprob *= 1 - width
                width = sum(widths) / max(1, len(widths))
                equity = self._range_equity(ix, width or .3)
                equity **= 1 + max(0., sum(widths) - .65) * .3
                rp = risk_premium(active, s.hero, s.payouts, min(1., eff / max(h.stack, 1)))
                ev = foldprob * pot + (1 - foldprob) * ((equity - rp) * (pot + 2 * eff - call) - eff)
                if ev > .08 * bb and (effective_percentile < opening + .10 or depth <= 4):
                    return Action("raise", s.max_raise_to)
                if call == 0:
                    return Action("check")
                if limpers and depth >= 7 and (r1 == r2 or suited) and call <= bb and call / (s.pot + call) < .16:
                    return Action("call")
                return Action("fold")
            if effective_percentile < opening:
                # Favor smaller opens late/short, charge limpers more.
                size = (2.15 if depth < 30 or nbehind <= 2 else 2.35) + .9 * limpers
                if s.hero in order[-2:] and limpers:
                    size += .7
                return Action("raise", round(size * bb))
            if call == 0:
                return Action("check")
            # Cheap overlimps with suited/paired hands, never speculative
            # calls against expensive raises without implied-odds depth.
            if limpers and depth >= 18 and call <= bb and (r1 == r2 or (suited and r1-r2 <= 3)) and call / (s.pot + call) < .22:
                return Action("call")
            if len(active) == 2 and call <= .5 * bb and percentile < .93:
                return Action("call")
            return Action("fold")

        last = raises[-1]
        raiser = next((p for p in opponents if p.seat == last.seat), opponents[0])
        t = self.stats[raiser.seat]
        rpos = order.index(raiser.seat)
        rbehind = len(order) - rpos - 1
        base_width = (.65, .55, .42, .30, .23, .19, .16, .14)[min(7, rbehind)]
        width = base_width * clamp(t.pfr / .18, .60, 1.8)
        if len(raises) >= 2:
            width = min(.18, .065 * clamp(t.pfr / .18, .6, 2.0))
        if len(raises) >= 3:
            width = min(width, .035 + .03 * max(0., t.pfr - .18))
        openbb = last.amount / bb
        if openbb >= 6 and depth > 20:
            width *= .65
        if raiser.all_in and openbb < 12 and len(raises) == 1:
            width = max(width, .16 + .016 * (12 - openbb))
        width = clamp(width, .025, .85)
        equity = self._range_equity(ix, width)
        callers = sum(p.seat != raiser.seat and p.committed >= last.amount and p.committed > bb for p in opponents)
        multi_equity = equity ** (1 + .65 * callers)
        price = call / max(1, pot + call)
        in_position = self._order(s).index(s.hero) > max(self._order(s).index(p.seat) for p in opponents)
        # Value three/four-bets and shallow reshoves. Mixing suited wheel-ace
        # bluffs only with depth, position and a credible opening range.
        premium = (r1 == r2 and r1 >= 10) or (r1 == 12 and r2 == 11)
        value_raise = equity > (.60 if len(raises) == 1 else .57) and percentile < (.055 if len(raises) == 1 else .025)
        if s.can_raise and h.stack > call:
            if depth <= 18 or call >= .28 * h.stack:
                if multi_equity > max(.47, price + .045 + risk) and (percentile < .17 or equity > .60):
                    return Action("raise", s.max_raise_to)
            elif value_raise or premium and len(raises) == 1:
                target = last.amount * (2.9 if in_position else 3.7) + callers * last.amount
                if target - h.committed > .40 * h.stack:
                    target = s.max_raise_to
                return Action("raise", round(target))
            elif len(raises) == 1 and not callers and suited and r1 == 12 and 0 <= r2 <= 3 and width >= .28 and depth >= 35 and self._random() < .16:
                return Action("raise", round(last.amount * (3.1 if in_position else 3.8)))
        if call == 0:
            return Action("check")
        realization = (0.91 if in_position else .81) + (.06 if suited else 0.)
        if call >= .8 * h.stack or all(p.all_in for p in opponents):
            realization = 1.
        elif depth <= 15:
            realization += .04
        # Small pairs can profit from deep, cheap set mining even though
        # their raw equity against a strong range is modest.
        if r1 == r2 and len(raises) == 1 and eff >= 16 * call and call <= 4 * bb and callers <= 2:
            return Action("call")
        if multi_equity * min(1., realization) >= price + risk + (.025 if len(raises) >= 2 else .018):
            # Avoid defending dominated weak offsuit hands with many players
            # still to act; the pot-odds model alone misses squeeze risk.
            if len(behind) >= 3 and not suited and r1 != r2 and percentile > .27:
                return Action("fold")
            return Action("call")
        return Action("fold")

    def _postflop(self, s, h, opponents):
        key = (s.hole, s.board)
        if key != self.cache_key:
            x = features(s.hole, s.board)
            es = predict(x)
            self.cache_key, self.cache = key, (x, es)
        x, es = self.cache
        street = s.street
        call = min(s.to_call, h.stack)
        eligible_cap = h.contributed + call
        pot = max(1, s.pot - sum(max(0, p.contributed - eligible_cap) for p in s.players)) if call == h.stack else max(1, s.pot)
        n = len(opponents)
        order = self._order(s)
        in_position = order.index(s.hero) > max(order.index(p.seat) for p in opponents)
        eff = max(0, min(h.stack, max(p.stack + p.committed - h.committed for p in opponents)))
        spr = eff / pot
        pre_raises = [e for e in s.history if e.street == "preflop" and e.action == "raise"]
        street_raises = [e for e in s.history if e.street == street and e.action == "raise"]
        previous_bets = [e for e in s.history if e.street not in ("preflop", street) and e.action == "raise" and e.seat != s.hero]
        aggressor = street_raises[-1].seat if street_raises else (pre_raises[-1].seat if pre_raises else None)
        t = self.stats.get(aggressor, Tendencies())
        pfr_width = clamp(t.pfr * 1.5, .08, .7)
        base = es[1] if not pre_raises else (es[2] if pfr_width >= .35 else .5 * (es[2] + es[3]))
        if len(pre_raises) >= 2:
            base = .65 * es[3] + .35 * es[4]
        # Model output is heads-up; exponent accounts for additional live
        # opponents. All-in/folded distinctions come from explicit state.
        current_strength = es[0] ** n
        base **= 1 + .8 * (n - 1)
        strong = es[6] ** (1 + .8 * (n - 1))
        nutrange = es[7] ** (1 + .8 * (n - 1))
        draw = len(s.board) < 5 and (x[23] == 4 and x[24] > 0 or x[37] >= 2)
        nut_flush_draw = len(s.board) < 5 and x[23] == 4 and x[24] >= 13
        wet = x[22] >= 2 and x[0] == 3 or x[29] >= 3 or x[22] >= 3
        active = [p for p in s.players if p.stack + p.contributed > 0]
        risk = risk_premium(active, s.hero, s.payouts, call / max(h.stack, 1))
        # Exact public-board ties: on an unbeatable board, share is 1/(n+1).
        # Quads with A kicker, royal flush, and A-high straight on rainbow
        # board cannot be beaten. Other board-only hands can still lose.
        locked = len(s.board) == 5 and (
            (x[7] == 8 and x[8] == 14) or
            (x[7] == 7 and (x[9] == 14 or x[8] == 14 and x[9] == 13)) or
            (x[7] == 4 and x[8] == 14 and x[22] < 3))
        if locked:
            eq = 1 / (n + 1)
            return Action("call" if call and eq * (pot + call) >= call else "fold" if call else "check")
        # Four public quads plus the highest possible private kicker cannot
        # lose, including on the turn. Random training sees this too rarely
        # to entrust an elimination decision to regression extrapolation.
        if x[7] == 7 and x[3] == (13 if x[8] == 14 else 14):
            if call:
                return Action("call")
            return Action("raise", h.committed + max(s.big_blind, round(pot * .5)))
        # A private royal flush is always the nuts. These rare states should
        # not depend on regression support in a random training population.
        absolute_nuts = x[1] == 8 and x[2] == 14
        if absolute_nuts:
            return Action("raise", s.max_raise_to if spr < 2 else h.committed + round(pot * .85))

        if call:
            bet_fraction = call / max(s.big_blind, pot - call)
            pressure = clamp((bet_fraction - .2) / .8, 0., 1.)
            eq = (1 - pressure) * es[5] + pressure * es[6]
            if street == "river":
                eq = (1 - .5 * pressure) * eq + .5 * pressure * es[7]
            if len(street_raises) >= 2 or previous_bets and street == "river":
                eq = .60 * eq + .40 * es[7]
            # Shrunk observed aggression/looseness softens overly narrow
            # priors against frequent bettors; passive large bets tighten.
            bluffiness = clamp((t.aggression - .33) * .8 + (t.looseness - .3) * .3, -.12, .30)
            if bluffiness >= 0:
                eq = eq * (1 - bluffiness) + es[1] * bluffiness
            else:
                eq = eq * (1 + bluffiness) - es[7] * bluffiness
            eq = .88 * eq + .12 * min(es[2], es[3]) if pre_raises else eq
            eq **= 1 + .85 * (n - 1)
            price = call / (pot + call)
            allin = h.stack <= call or all(p.stack == 0 for p in opponents)
            realization = 1. if street == "river" or allin else (.96 if in_position else .89)
            if draw:
                realization = min(1., realization + .04)
            margin = .012 if allin or street == "river" else .018
            # Strong value raises, constrained by the equity of the range
            # likely to continue. Weak board-only pairs never get this bonus.
            can_value = strong > (.69 if street == "river" else .64) and current_strength > .82
            if s.can_raise and h.stack > call and (can_value or spr < .65 and eq > .61):
                target = h.committed + call + round((pot + call) * (.70 if wet else .58))
                if target - h.committed > .48 * h.stack or spr < 1.2:
                    target = s.max_raise_to
                return Action("raise", target)
            # Selective semi-bluffs against one opponent, with real outs,
            # affordable pressure, and evidence that folds exist.
            if n == 1 and draw and nut_flush_draw and s.can_raise and spr > 2 and len(street_raises) == 1 and t.fold_rate > .43 and eq > .38 and self._random() < .14:
                return Action("raise", h.committed + call + round((pot + call) * .7))
            if eq * realization >= price + risk + margin:
                return Action("call")
            # Modest implied-odds credit for high flush draws, never on river
            # or against an already all-in opponent.
            if nut_flush_draw and not allin and n <= 2 and call <= .20 * eff and eq + .025 >= price + risk:
                return Action("call")
            return Action("fold")

        if all(p.stack == 0 for p in opponents):
            return Action("check")
        # Value sizing grows with texture and hand strength. Against several
        # opponents, thin one-pair bets and automatic continuation bets vanish.
        value = strong >= (.55 if street == "river" else .53) and current_strength > (.62 if n == 1 else .74)
        if value:
            size = .68 if wet else .52
            if nutrange > .72:
                size = .90 if street == "river" else .76
            elif strong < .62:
                size = .38
            # Do not slowplay low-SPR hands; avoid tiny bets that strand chips.
            amount = max(s.big_blind, round(pot * size))
            if amount >= .48 * eff and base > .65:
                return Action("raise", min(s.max_raise_to, h.committed + eff))
            return Action("raise", h.committed + amount)
        is_aggressor = bool(pre_raises and pre_raises[-1].seat == s.hero)
        last_own = any(e.seat == s.hero and e.action == "raise" and e.street not in ("preflop", street) for e in s.history)
        foldrate = sum(self.stats[p.seat].fold_rate for p in opponents) / n
        bluff_freq = 0.
        size = .33
        if n == 1:
            if street == "flop" and is_aggressor:
                bluff_freq = .47 if not wet else .24
                if draw:
                    bluff_freq += .15
            elif street == "turn" and draw and (last_own or in_position):
                bluff_freq = .42
                size = .62
            elif in_position and not street_raises and not previous_bets and street != "river":
                bluff_freq = .28 if base > .32 else .16
            elif street == "river" and last_own and current_strength < .25 and x[13] >= 12:
                # Blocker-aware low-frequency river continuation bluff.
                bluff_freq = .15
                size = .68
            bluff_freq *= clamp(foldrate / .4, .45, 1.6)
            if x[42] or spr < .8:
                bluff_freq = 0.
        if bluff_freq and self._random() < bluff_freq:
            return Action("raise", h.committed + max(s.big_blind, round(pot * size)))
        # Protection with vulnerable medium strength in position.
        if n == 1 and street != "river" and in_position and current_strength > .68 and base > .57 and self._random() < .35:
            return Action("raise", h.committed + max(s.big_blind, round(pot * .38)))
        return Action("check")
