"""Position-aware, range-conditioned tournament NLHE strategy."""
from dataclasses import dataclass
import random
from time import perf_counter_ns
from .cards import hand_class
from .equity import PREFLOP, preflop_equity, equity, features
from .icm import call_threshold
from .interface import Action, Observation, legalize, passive


@dataclass(slots=True)
class Opponent:
    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    faced: int = 0
    folded: int = 0
    range_fraction: float = 1.0
    aggression: int = 0
    entered: bool = False
    raised: bool = False

    def looseness(self):
        return (self.vpip + 8.0) / (self.hands + 24.0)

    def raising(self):
        return (self.pfr + 4.5) / (self.hands + 24.0)

    def folding(self):
        return (self.folded + 5.0) / (self.faced + 12.0)


class Agent:
    """One instance per table. No cross-table opponent identity or persistence.

    act(Observation) -> Action is the timed local contract. JSON conversion and
    validation are separate development conveniences, measured independently.
    """
    def __init__(self, seed=7007, comparisons=48, budget_us=450):
        self.seed = seed
        self.comparisons = comparisons
        self.budget_ns = int(budget_us * 1000) if budget_us else None
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.table = None
        self.hand = None
        self.street = None
        self.last_event = -1
        self.models = {}
        self.preflop_raises = 0
        self.aggressor = None
        self.errors = 0

    def _observe(self, obs):
        if self.table != obs.table_id:
            self.reset()
            self.table = obs.table_id
        new_hand = self.hand != obs.hand_id
        if new_hand:
            self.hand = obs.hand_id
            self.street = obs.street
            self.preflop_raises = 0
            self.aggressor = None
            for p in obs.players:
                m = self.models.setdefault(p.seat, Opponent())
                m.range_fraction = 1.0
                m.aggression = 0
                m.entered = m.raised = False
                if p.stack + p.committed > 0:
                    m.hands += 1
        elif self.street != obs.street:
            self.street = obs.street
            for m in self.models.values():
                m.aggression = 0
        for e in obs.events[-32:]:
            if e.id <= self.last_event:
                continue
            self.last_event = e.id
            m = self.models.setdefault(e.seat, Opponent())
            if e.to_call > 0:
                m.faced += 1
                m.folded += e.action == "fold"
            if e.street == "preflop":
                if e.action in ("call", "raise") and not m.entered:
                    m.vpip += 1
                    m.entered = True
                if e.action == "raise":
                    if not m.raised:
                        m.pfr += 1
                        m.raised = True
                    self.preflop_raises += 1
                    self.aggressor = e.seat
                    if self.preflop_raises == 1:
                        m.range_fraction = min(.55, max(.12, m.raising() * 1.4))
                    elif self.preflop_raises == 2:
                        m.range_fraction = min(.16, max(.045, m.raising() * .40))
                    else:
                        m.range_fraction = min(.08, max(.025, m.raising() * .18))
                elif e.action == "call":
                    m.range_fraction = min(m.range_fraction, max(.18, m.looseness()) if self.preflop_raises else max(.45, m.looseness()))
            elif e.street == obs.street:
                if e.action == "raise":
                    m.aggression = 2 if e.to_call else 1
                elif e.action == "call":
                    m.aggression = max(m.aggression, 1)

    @staticmethod
    def _position(obs, me):
        active = sorted(p.seat for p in obs.players if p.stack + p.committed > 0)
        order = sorted(active, key=lambda seat: (seat - obs.button - 1) % 8)
        pre = order[2:] + order[:2] if len(order) > 2 else list(reversed(order))
        behind = sum(not next(p for p in obs.players if p.seat == s).folded for s in pre[pre.index(me.seat) + 1:])
        live = [s for s in order if not next(p for p in obs.players if p.seat == s).folded]
        return behind, bool(live and live[-1] == me.seat)

    def _risk(self, obs, me, opponents, pot):
        if not obs.payouts or len(opponents) != 1 or obs.to_call < .15 * (me.stack + me.bet):
            return 0.0
        if pot != obs.pot:
            return 0.0
        # Exact ICM over chips behind, using a clearly stated heads-up fold
        # allocation approximation. Avoid the computation in routine small pots.
        players = [p for p in obs.players if p.stack + p.committed > 0]
        hero = next(i for i, p in enumerate(players) if p.seat == me.seat)
        villain = next(i for i, p in enumerate(players) if p.seat == opponents[0].seat)
        threshold = call_threshold([p.stack for p in players], hero, villain,
                                   obs.pot, obs.to_call, obs.payouts)
        return max(0.0, threshold - obs.to_call / (pot + obs.to_call))

    @staticmethod
    def _eligible_pot(obs, me):
        if obs.eligible_pot is not None:
            return obs.eligible_pot
        cap = me.committed + obs.to_call
        return sum(min(p.committed, cap) for p in obs.players)

    @staticmethod
    def _pot_groups(obs, me, opponents):
        """Actual pot tiers, with still-active players eligible to match bets."""
        if not any(p.stack == 0 for p in opponents):
            return None
        cap = me.committed + obs.to_call
        contributions = [min(cap, p.committed + (obs.to_call if p.seat == me.seat else 0)) for p in obs.players]
        groups = []
        previous = 0
        for level in sorted(set(x for x in contributions if x > 0)):
            amount = (level - previous) * sum(x >= level for x in contributions)
            eligible = tuple(i for i, p in enumerate(opponents) if p.stack > 0 or p.committed >= level)
            if groups and groups[-1][1] == eligible:
                groups[-1] = (groups[-1][0] + amount, eligible)
            else:
                groups.append((amount, eligible))
            previous = level
        return groups if len(groups) > 1 else None

    def _preflop(self, obs, me, opponents, behind, in_position):
        a, b = obs.hole
        entry = PREFLOP[hand_class(a, b)]
        rank_a, rank_b = a // 4 + 2, b // 4 + 2
        pair = rank_a == rank_b
        suited = a % 4 == b % 4
        gap = abs(rank_a - rank_b)
        percentile = entry["percentile"]
        bb = obs.big_blind
        effective = min(me.stack + me.bet, max((p.stack + p.bet for p in opponents), default=me.stack))
        depth = effective / bb
        pot = self._eligible_pot(obs, me)
        cost = obs.to_call
        current = max((p.bet for p in obs.players), default=bb)
        raised = current > bb
        can_raise = "raise" in obs.legal_actions
        if not raised:
            limpers = sum(self.models[p.seat].entered for p in opponents)
            # HU button opens broadly; early eight-handed seats remain selective.
            threshold = (.60 if len(opponents) == 1 else
                         .46 if behind <= 1 else .39 if behind == 2 else .28 if behind == 3
                         else .22 if behind == 4 else .18 if behind == 5 else .16)
            playable = percentile
            if depth > 20:
                playable -= .035 * pair + (.045 if suited and gap <= 2 else .015 if suited else 0)
            threshold -= min(.08, .025 * limpers)
            if depth <= 12 and can_raise:
                # Compare a shove to folding, assuming at most one principal
                # caller. Multi-caller risk is compensated by a margin.
                shove = min(me.stack, max(0, effective - me.bet))
                calling = min(.25, max(.08, .08 + .9 / max(depth, 2)))
                folds = (1.0 - calling) ** max(1, behind + limpers)
                called_eq = preflop_equity(a, b, calling)
                paid = max(0, shove + me.bet - bb)
                ev = folds * pot + (1 - folds) * (called_eq * (pot + paid) - (1 - called_eq) * shove)
                if ev > bb * (.10 + .08 * max(0, behind - 1)) and playable < min(.75, threshold + .16):
                    return Action("raise", obs.max_raise_to)
            if playable < threshold and can_raise:
                size = (2.15 + .85 * limpers + (.35 if behind <= 1 and len(opponents) > 1 else 0)) * bb
                if size >= .35 * (me.stack + me.bet):
                    size = obs.max_raise_to
                return Action("raise", round(size))
            if cost == 0:
                return Action("check")
            # Complete the small blind into several limpers with playable hands;
            # marginal offsuit hands do not defend merely because of sunk chips.
            if limpers and cost <= .5 * bb and playable < .72:
                return Action("call")
            return Action("fold")

        contenders = [p for p in opponents if p.bet > bb or self.models[p.seat].entered]
        primary = max(opponents, key=lambda p: p.bet)
        model = self.models[primary.seat]
        fraction = model.range_fraction
        if fraction >= .99:
            fraction = .22 if current <= 4 * bb else .09 if current <= 12 * bb else .045
        eq = preflop_equity(a, b, fraction)
        n = max(1, len(contenders))
        # Fractional exponent reflects correlated outcomes against multiple
        # ranges; random multiway equity provides a second conservative anchor.
        groups = self._pot_groups(obs, me, opponents)
        if groups:
            def group_equity(count):
                return min(eq ** (1 + .68 * (count - 1)), entry["multiway"][min(count, 7) - 1] + .04) if count else 1.0
            eq = sum(amount * group_equity(len(eligible)) for amount, eligible in groups) / sum(amount for amount, _ in groups)
        else:
            eq = min(eq ** (1 + .68 * (n - 1)), entry["multiway"][min(n, 7) - 1] + .04)
        price = cost / max(1, pot + cost)
        risk = self._risk(obs, me, opponents, pot)
        all_in = cost >= .90 * me.stack or all(p.stack == 0 for p in contenders or opponents)
        realization = 1.0 if all_in else (.94 if in_position else .87)
        adjusted = eq * realization - .012 * max(0, behind - 1)
        reraise = self.preflop_raises >= 2 or current > 5 * bb
        premium = pair and rank_a >= 13
        value_raise = ((premium and eq > .52 + risk) or (percentile < (.035 if not reraise else .018) and eq > .55 + risk))
        if can_raise and value_raise:
            size = current * (2.8 if in_position else 3.5) + max(0, len(contenders) - 1) * current * .75
            if depth <= 22 or size >= .34 * (me.stack + me.bet):
                size = obs.max_raise_to
            return Action("raise", round(size))
        # A few suited-ace blocker 3-bets when there is room to fold to a 4-bet.
        if can_raise and not reraise and depth >= 35 and n == 1 and suited and max(rank_a, rank_b) == 14 and min(rank_a, rank_b) <= 5:
            if self.rng.random() < .12 * (1.0 + model.folding()):
                return Action("raise", round(current * (3 if in_position else 3.6)))
        if adjusted >= price + risk + (.025 if reraise and not all_in else .012):
            return Action("call") if cost else Action("check")
        # Limited implied-odds calls: no set mining against a short stack.
        if not reraise and n <= 3 and depth >= 25 and cost <= .055 * effective and pair and price < .32:
            return Action("call")
        if not reraise and in_position and suited and gap <= 2 and depth >= 40 and cost <= .035 * effective and price < .27:
            return Action("call")
        return passive(obs)

    def _postflop(self, obs, me, opponents, in_position, deadline):
        pot = self._eligible_pot(obs, me)
        cost = obs.to_call
        n = len(opponents)
        ranges = []
        for p in opponents:
            m = self.models[p.seat]
            fraction = m.range_fraction
            # If no action history was delivered, infer a conservative betting
            # range from current public action; never infer hidden cards.
            aggr = m.aggression
            if p.bet > me.bet:
                aggr = max(aggr, 2 if p.bet >= pot * .65 else 1)
            ranges.append((fraction, aggr))
        risk = self._risk(obs, me, opponents, pot) if cost else 0.0
        eq = equity(obs.hole, obs.board, ranges, self.rng, self.comparisons, deadline,
                    self._pot_groups(obs, me, opponents))
        made, draw, wet = features(obs.hole, obs.board)
        can_raise = "raise" in obs.legal_actions
        remaining = len(obs.board) < 5
        effective = min(me.stack, max((p.stack + max(0, p.bet - me.bet) for p in opponents), default=0))
        spr = effective / max(1, pot)
        if cost:
            price = cost / max(1, pot + cost)
            realization = 1.0
            if remaining and cost < .75 * me.stack and made < .6:
                realization = .94 if in_position else .88
            adjusted = eq * realization
            # Draws with insufficient direct odds need depth and position to
            # earn modest implied odds; never grant them against all-in bets.
            implied = min(.035, draw * .08) if remaining and in_position and effective > 4 * cost else 0.0
            raise_value = eq > (.74 if n == 1 else .70) + min(.10, risk) and made > .60
            if can_raise and raise_value:
                extra = .72 * (pot + cost)
                target = me.bet + cost + max(extra, cost)
                if target - me.bet >= .55 * me.stack or spr <= 1.1:
                    target = obs.max_raise_to
                return Action("raise", round(target))
            if can_raise and n == 1 and draw > .37 and remaining and cost < .25 * pot and spr > 2 and risk < .03:
                if self.rng.random() < .13 and eq > price:
                    return Action("raise", round(me.bet + cost + .8 * (pot + cost)))
            if adjusted + implied >= price + risk + .018:
                return Action("call")
            return passive(obs)

        if not can_raise or not opponents:
            return Action("check")
        # Value sizing scales with range equity and board texture. Strong hands
        # deny draw equity; small bets retain dominated one-pair hands.
        if eq > (.60 if n == 1 else .53) and made >= (.48 if n == 1 else .59):
            fraction = .38 if made < .7 and wet < .65 else .68
            if eq > .83 and made > .8:
                fraction = .95
            amount = max(obs.big_blind, pot * fraction)
            if spr < 1.2 and eq > .67:
                amount = me.stack
            return Action("raise", round(me.bet + amount))
        if n == 1:
            model = self.models[opponents[0].seat]
            fold = model.folding()
            initiative = self.aggressor == me.seat
            chance = 0.0
            amount = pot * .48
            if remaining and draw >= .25 and eq > .25:
                chance = .45 if in_position else .27
                amount = pot * .62
            elif len(obs.board) == 3 and initiative and wet < .74:
                chance = .42 if in_position else .25
                amount = pot * .33
            elif in_position and made < .25:
                # Restrained river bluffs prefer ace / high flush blockers.
                blocker = any(c // 4 == 12 for c in obs.hole)
                chance = (.18 if blocker else .08) if not remaining else .15
                amount = pot * .65
            chance *= min(1.5, max(.25, fold / .42))
            if self.rng.random() < chance and spr > .7:
                return Action("raise", round(me.bet + max(obs.big_blind, amount)))
        elif n == 2 and in_position and draw > .43 and eq > .35 and self.rng.random() < .2:
            return Action("raise", round(me.bet + pot * .60))
        return Action("check")

    def act(self, obs):
        """Timed entry point. Input is an already normalized Observation."""
        deadline = perf_counter_ns() + self.budget_ns if self.budget_ns else None
        try:
            self._observe(obs)
            me = next(p for p in obs.players if p.seat == obs.hero)
            opponents = [p for p in obs.players if p.seat != obs.hero and not p.folded and p.stack + p.committed > 0]
            if not opponents:
                return passive(obs)
            behind, position = self._position(obs, me)
            if not obs.board:
                action = self._preflop(obs, me, opponents, behind, position)
            else:
                action = self._postflop(obs, me, opponents, position, deadline)
            return legalize(obs, action)
        except (ValueError, KeyError, IndexError, ZeroDivisionError, StopIteration, TypeError):
            self.errors += 1
            return passive(obs)

    def act_dict(self, value):
        return self.act(Observation.from_dict(value)).as_dict()

