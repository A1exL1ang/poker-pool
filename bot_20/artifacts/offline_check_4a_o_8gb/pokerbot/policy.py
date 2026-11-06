"""Small, auditable range-aware NLHE policy; not an equilibrium solver.

The learned component estimates showdown shares only. Strategic decisions use
pot odds, position, legal raises, bounded anonymous-seat adaptation, and actual
provided payouts. No online service, solver, file write, or Monte Carlo in act().
"""

from dataclasses import dataclass
from math import prod
from random import Random
from time import perf_counter_ns

from .interface import call, passive, raise_to
from .icm import icm_hero


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


@dataclass(frozen=True)
class PolicyConfig:
    open_scale: float = 1.0
    call_margin: float = 0.025
    value_margin: float = 0.075
    bluff_scale: float = 1.0
    pressure_scale: float = 1.0
    shove_limit_bb: float = 10.0
    risk_scale: float = 1.0
    adaptation: bool = True
    budget_ns: int = 700_000


class SeatStats:
    __slots__ = ("pf_seen", "vpip", "raised", "post_seen", "post_agg", "faced", "folded")

    def __init__(self):
        self.pf_seen = self.vpip = self.raised = 0
        self.post_seen = self.post_agg = self.faced = self.folded = 0

    def looseness(self):
        return (5.6 + self.vpip) / (20.0 + self.pf_seen)

    def aggression(self):
        return (5.0 + self.post_agg) / (16.0 + self.post_seen)

    def fold_rate(self):
        return (7.0 + self.folded) / (16.0 + self.faced)


class Agent:
    """Local Agent. Construct/load before the timed decision interval.

    reset() must be called for a new table, or provide a changing table_id.
    A decreasing hand_id also triggers reset. State never persists on disk.
    """

    def __init__(self, seed=20, config=None):
        from .equity import EquityEstimator
        from .selected import SELECTED_CONFIG
        self.estimator = EquityEstimator()
        self.config = config if config is not None else PolicyConfig(**SELECTED_CONFIG)
        self.seed = seed
        self.rng = Random(seed)
        self.decisions = self.errors = self.budget_fallbacks = 0
        self.reset()

    def reset(self, table_id=None):
        self.rng = Random(self.seed)
        self.table_id = table_id
        self.last_hand = -1
        self.history_count = 0
        self.pf_seen_this_hand = set()
        self.stats = [SeatStats() for _ in range(8)]
        self.feature_key = None
        self.features = None

    def act(self, obs):
        start = perf_counter_ns()
        self.decisions += 1
        try:
            self._observe(obs)
            if obs["street"] == 0:
                return self._preflop(obs, start)
            return self._postflop(obs, start)
        except (KeyError, IndexError, TypeError, ValueError, ArithmeticError):
            self.errors += 1
            return passive(obs)

    decide = act

    def _observe(self, obs):
        hand = obs["hand_id"]
        tid = obs.get("table_id", self.table_id)
        if tid != self.table_id or hand < self.last_hand:
            self.reset(tid)
        if hand != self.last_hand:
            self.last_hand = hand
            self.history_count = 0
            self.pf_seen_this_hand.clear()
            self.feature_key = None
        history = obs["history"]
        if not self.config.adaptation:
            self.history_count = len(history)
            return
        # Only observed opportunities count; unobserved final actions are not
        # invented. Priors deliberately dominate small, selection-biased samples.
        for ev in history[self.history_count:]:
            seat, action = ev["seat"], ev["action"]
            if seat == obs["hero"] or action not in ("fold", "check", "call", "raise"):
                continue
            st = self.stats[seat]
            if ev["street"] == 0:
                if seat not in self.pf_seen_this_hand:
                    self.pf_seen_this_hand.add(seat)
                    if action != "check":
                        st.pf_seen += 1
                        st.vpip += action in ("call", "raise")
                        st.raised += action == "raise"
            else:
                st.post_seen += action != "fold"
                st.post_agg += action == "raise"
                if ev.get("to_call", 0) > 0:
                    st.faced += 1
                    st.folded += action == "fold"
        self.history_count = len(history)

    @staticmethod
    def _live(obs):
        return [i for i, f in enumerate(obs["folded"]) if not f and i != obs["hero"]]

    @staticmethod
    def _in_position(obs, opponents):
        n, b = len(obs["stacks"]), obs["button"]
        # Postflop first seat left of button, button last. All-ins cannot act.
        hero_order = (obs["hero"] - b - 1) % n
        return all(obs["stacks"][i] == 0 or (i - b - 1) % n < hero_order for i in opponents)

    @staticmethod
    def _behind(obs):
        """Seats still to act in normal preflop order, including the blinds."""
        n, h = len(obs["stacks"]), obs["hero"]
        alive = [i for i, s in enumerate(obs["hand_start_stacks"]) if s > 0]
        first = obs["sb_seat"] if len(alive) == 2 else (obs["bb_seat"] + 1) % n
        hero_order = (h - first) % n
        return [i for i in alive if i != h and not obs["folded"][i]
                and obs["stacks"][i] > 0 and (i - first) % n > hero_order]

    def _ranges(self, obs, opponents):
        ranges = {i: 1.0 for i in opponents}
        raisers = []
        post_bets = {i: [] for i in opponents}
        bet_level = [0, 0, 0, 0]
        for ev in obs["history"]:
            i, a, street = ev["seat"], ev["action"], ev["street"]
            increment = max(0, ev["amount"] - bet_level[street]) if a == "raise" else 0
            if a == "raise":
                bet_level[street] = ev["amount"]
            if street == 0 and a == "raise":
                raisers.append(i)
            if i not in ranges:
                continue
            if street == 0:
                looseness = self.stats[i].looseness()
                if a == "raise":
                    nr = len(raisers)
                    if nr == 1:
                        n = len(obs["stacks"])
                        offset = (i - obs["bb_seat"] - 1) % n
                        position = sum(obs["hand_start_stacks"][(obs["bb_seat"] + 1 + j) % n] > 0
                                       for j in range(offset))
                        occupied = sum(x > 0 for x in obs["hand_start_stacks"])
                        base = 0.16 + 0.045 * position
                        if occupied == 2:
                            base = 0.60
                        elif i == obs["button"]:
                            base = 0.44
                        elif i == obs["sb_seat"]:
                            base = 0.51
                        size = ev["amount"] / obs["big_blind"]
                        if size > 5:
                            base *= max(0.48, 1.0 - 0.025 * (size - 5))
                        ranges[i] = clamp(base * (looseness / 0.28) ** 0.7, 0.06, 0.9)
                    else:
                        ranges[i] = clamp((0.075 if nr == 2 else 0.032)
                                          * (looseness / 0.28) ** 1.1, 0.025, 0.4)
                elif a == "call":
                    ranges[i] = min(ranges[i], clamp(looseness * (0.83 if raisers else 1.3), 0.10, 0.85))
            elif a == "raise":
                size = increment / max(obs["big_blind"], ev.get("pot_before", 0))
                post_bets[i].append((street, size))
        pressure = 0.0
        aggressor = None
        for i, bets in post_bets.items():
            if not bets:
                continue
            street, size = bets[-1]
            p = 0.25 + 0.09 * street + 0.11 * min(2.0, size)
            p += 0.12 * (len(bets) - 1)
            if sum(s == street for s, _ in bets) > 1:
                p += 0.12
            p *= clamp(0.32 / self.stats[i].aggression(), 0.60, 1.28)
            if p > pressure:
                pressure, aggressor = p, i
        if not ranges:
            return 1.0, 0.0, raisers, aggressor
        # Tightest player dominates; retain some weight from the full field.
        values = list(ranges.values())
        rf = 0.65 * min(values) + 0.35 * (sum(values) / len(values))
        return rf, clamp(pressure * self.config.pressure_scale, 0.0, 1.0), raisers, aggressor

    @staticmethod
    def _eligible_pot(obs, cost):
        cap = obs["committed"][obs["hero"]] + cost
        return sum(min(c, cap) for c in obs["committed"]) + cost

    def _risk_premium(self, obs, cost, opponents):
        payouts = obs.get("payouts")
        if not payouts or self.config.risk_scale == 0:
            return 0.0
        alive = [obs["stacks"][i] + obs["committed"][i] for i in range(len(obs["stacks"]))
                 if obs["hand_start_stacks"][i] > 0]
        m = len(alive)
        if m <= 2 or payouts[1] == payouts[m - 1]:
            return 0.0
        p = payouts[:m]
        span = p[0] - p[-1]
        if span <= 0:
            return 0.0
        paid = sum(x > p[-1] for x in p)
        h = obs["hero"]
        stack = obs["stacks"][h] + obs["committed"][h]
        avg = sum(alive) / m
        broad = sum(x - p[-1] for x in p[1:]) / max(span, sum(x - p[-1] for x in p))
        bubble = 1.6 if m == paid + 1 else 1.1
        medium = clamp(avg / max(1, stack), 0.45, 1.25)
        covered = any(obs["stacks"][i] + obs["committed"][i] >= stack for i in opponents)
        fraction = cost / max(1, obs["stacks"][h])
        return clamp(0.18 * broad * bubble * medium * (0.3 + 0.7 * fraction)
                     * (1 if covered else 0.55) * self.config.risk_scale, 0, 0.16)

    def _call_threshold(self, obs, cost, opponents, terminal=False):
        total = self._eligible_pot(obs, cost)
        price = cost / max(1, total)
        payouts = obs.get("payouts")
        seats = [i for i, s in enumerate(obs["hand_start_stacks"]) if s > 0]
        # A nearly committing call is priced as a binary showdown for ICM,
        # even if the bettor retains a few chips. This is a risk approximation
        # for future streets, not a claim that betting has actually ended.
        terminal = terminal or cost >= 0.5 * obs["stacks"][obs["hero"]]
        if terminal and len(opponents) == 1 and payouts and len(seats) <= 8 and self.config.risk_scale:
            h, v = obs["hero"], opponents[0]
            hero = seats.index(h)
            villain = seats.index(v)
            base = [obs["stacks"][i] for i in seats]
            fold_stacks = base.copy()
            fold_stacks[villain] += obs["pot"]
            lose_stacks = base.copy()
            lose_stacks[hero] -= cost
            lose_stacks[villain] += obs["pot"] + cost
            win_stacks = base.copy()
            win_stacks[hero] += total - cost
            win_stacks[villain] += obs["pot"] + cost - total
            pf = icm_hero(fold_stacks, payouts[:len(seats)], hero)
            pl = icm_hero(lose_stacks, payouts[:len(seats)], hero)
            pw = icm_hero(win_stacks, payouts[:len(seats)], hero)
            if pw > pl + 1e-12:
                icm_price = clamp((pf - pl) / (pw - pl), 0, 1)
                return clamp(price + self.config.risk_scale * (icm_price - price), 0, 1)
        return min(0.98, price + self._risk_premium(obs, cost, opponents))

    def _raise_is_safe(self, obs, target, opponents, equity):
        """Compare a large called raise with folding under supplied ICM payouts.

        Re-express the opponent's hypothetical matching chips as a bet being
        faced. This makes _call_threshold's fold/win/loss stacks conserve chips.
        It is a binary-showdown approximation, not a full future-street solve.
        """
        if not obs.get("payouts") or not self.config.risk_scale or len(opponents) != 1:
            return True
        h, v = obs["hero"], opponents[0]
        actual = raise_to(obs, target)
        if actual["action"] != "raise":
            return True
        effective_to = min(actual["amount"], obs["street_committed"][v] + obs["stacks"][v])
        cost = max(0, effective_to - obs["street_committed"][h])
        if cost < 0.15 * max(1, obs["stacks"][h]):
            return True
        other_cost = max(0, effective_to - obs["street_committed"][v])
        adjusted = dict(obs)
        adjusted["stacks"] = obs["stacks"].copy()
        adjusted["committed"] = obs["committed"].copy()
        adjusted["stacks"][v] -= other_cost
        adjusted["committed"][v] += other_cost
        adjusted["pot"] = obs["pot"] + other_cost
        return equity >= self._call_threshold(adjusted, cost, opponents, True)

    def _preflop(self, obs, start):
        from .equity import preflop_percentile
        h, bb = obs["hero"], obs["big_blind"]
        stack = obs["stacks"][h]
        cost = min(stack, obs["to_call"])
        own = obs["street_committed"][h]
        opponents = self._live(obs)
        if not opponents:
            return call(obs)
        hole = obs["hole_cards"]
        ranks = sorted((c // 4 + 2 for c in hole), reverse=True)
        suited = hole[0] % 4 == hole[1] % 4
        pair = ranks[0] == ranks[1]
        pct = preflop_percentile(hole)
        behind = self._behind(obs)
        rf, _, raisers, _ = self._ranges(obs, opponents)
        if not obs["can_raise"] and (cost >= stack or all(obs["stacks"][i] == 0 for i in opponents)):
            equity = self.estimator.estimate(hole, [], len(opponents), rf, 0)
            threshold = self._call_threshold(obs, cost, opponents, True)
            return call(obs) if cost == 0 or equity >= threshold else passive(obs)
        limpers = sum(ev["street"] == 0 and ev["action"] == "call" for ev in obs["history"])
        effective = min(stack + own, max(obs["stacks"][i] + obs["street_committed"][i] for i in opponents))
        eff_bb = effective / bb
        unopened = not raisers
        ip = self._in_position(obs, opponents)
        if unopened:
            # Percentile comes from independent native Monte Carlo preflop data.
            width = (0.58, 0.46, 0.30, 0.24, 0.20, 0.17, 0.145, 0.13)[min(7, len(behind))]
            if h == obs["sb_seat"]:
                width = 0.58 if len(opponents) > 1 else 0.80
            if h == obs["bb_seat"]:
                width = 0.30
            width *= self.config.open_scale
            width /= 1 + 0.20 * limpers
            if suited and ranks[0] - ranks[1] <= 3 and eff_bb >= 25:
                width += 0.045
            if pair and eff_bb >= 30:
                width += 0.025
            if obs["can_raise"] and eff_bb <= self.config.shove_limit_bb:
                number = max(1, len(opponents))
                call_range = clamp(0.07 + 0.16 / number ** 0.65
                                   + (0.05 if number == 1 else 0)
                                   + 0.04 * (8 / max(2, eff_bb) - 1), 0.07, 0.60)
                probs = [1.0 if obs["stacks"][i] == 0 else
                         clamp(call_range * (self.stats[i].looseness() / 0.28) ** 0.65
                               * (1.2 if limpers else 1), 0.05, 0.7)
                         for i in opponents]
                fold_prob = prod(1 - p for p in probs)
                n_allin = sum(obs["stacks"][i] == 0 for i in opponents)
                eq = self.estimator.estimate(hole, [], max(1, n_allin), min(rf, call_range), 0)
                risk = max(0, effective - own)
                matched = sum(p * min(obs["stacks"][i], max(0, effective - obs["street_committed"][i]))
                              for p, i in zip(probs, opponents)) / max(1e-9, sum(probs))
                premium = self._risk_premium(obs, risk, opponents)
                # Single caller model, with a multi-caller uncertainty deduction.
                eq -= min(0.055, max(0, sum(probs) - 0.35) * 0.055) + premium
                ev = fold_prob * obs["pot"] + (1 - fold_prob) * (eq * (obs["pot"] + risk + matched) - risk)
                if (ev > 0.10 * bb or pct < 0.025) and self._raise_is_safe(obs, obs["max_raise_to"], opponents, eq):
                    return raise_to(obs, obs["max_raise_to"])
                return call(obs) if cost == 0 else passive(obs)
            if pct <= width and obs["can_raise"]:
                size = (2.2 if eff_bb < 35 else 2.5) + min(4, limpers)
                if limpers and ranks[0] >= 12:
                    size += 0.4
                target = size * bb
                if target >= 0.42 * (stack + own) and pct < min(width, 0.22):
                    target = obs["max_raise_to"]
                if not self._raise_is_safe(obs, target, opponents, self.estimator.estimate(hole, [], 1, 0.16, 0)):
                    return call(obs) if cost == 0 else passive(obs)
                return raise_to(obs, target)
            if cost == 0:
                return call(obs)
            # Complete very cheap multiway entries; avoid large speculative calls.
            if cost <= 0.55 * bb and (pct < 0.66 or suited) and eff_bb > 12:
                return call(obs)
            if limpers and eff_bb > 22 and (pair or (suited and ranks[0] - ranks[1] <= 2)):
                if cost <= bb and pct < 0.55:
                    return call(obs)
            return passive(obs)
        # Only voluntary continuing opponents are counted for a decision before
        # others act; an explicit cushion covers squeeze/future-call uncertainty.
        involved = {ev["seat"] for ev in obs["history"] if ev["street"] == 0
                    and ev["action"] in ("call", "raise") and ev["seat"] in opponents}
        count = max(1, len(involved))
        equity = self.estimator.estimate(hole, [], count, rf, 0)
        terminal = cost >= stack or all(obs["stacks"][i] == 0 for i in opponents)
        threshold = self._call_threshold(obs, cost, opponents, terminal)
        if not terminal:
            realization = 0.93 if ip else 0.85
            if suited:
                realization += 0.035
            if pair:
                realization += 0.025
            equity *= min(0.98, realization)
            threshold += self.config.call_margin + (0.025 if len(raisers) >= 2 else 0)
        if obs["can_raise"]:
            premium_hand = pct < (0.033 if len(raisers) == 1 else 0.012)
            value = equity > (0.60 if count == 1 else 0.47)
            if (premium_hand or value) and equity >= threshold:
                target = (own + obs["to_call"]) * (3.0 if ip else 3.7) + max(0, count - 1) * bb
                if target >= 0.36 * (own + stack) or eff_bb < 18:
                    target = obs["max_raise_to"]
                if self._raise_is_safe(obs, target, opponents, equity):
                    return raise_to(obs, target)
            if (len(raisers) == 1 and count == 1 and suited and ranks[0] == 14 and ranks[1] <= 5
                    and eff_bb > 35 and rf > 0.28 and self.rng.random() < 0.12 * self.config.bluff_scale):
                return raise_to(obs, (own + obs["to_call"]) * (3.0 if ip else 3.7))
        return call(obs) if equity >= threshold else passive(obs)

    def _postflop(self, obs, start):
        from .features import extract_features
        h, stack = obs["hero"], obs["stacks"][obs["hero"]]
        opponents = self._live(obs)
        if not opponents:
            return call(obs)
        n = len(opponents)
        hole, board = obs["hole_cards"], obs["board"]
        if self._unbeatable_board(board):
            # Every eligible pot is necessarily shared. Calling cannot lose
            # chips; a win/loss ICM approximation is inappropriate for this case.
            return call(obs)
        key = (tuple(hole), tuple(board))
        if key != self.feature_key:
            self.features = extract_features(hole, board)
            self.feature_key = key
        features = self.features
        rf, pressure, raisers, aggressor = self._ranges(obs, opponents)
        eq = self.estimator.predict(features, n, rf, pressure)
        cost = min(stack, obs["to_call"])
        ip = self._in_position(obs, opponents)
        terminal = obs["street"] == 3 or cost >= stack or all(obs["stacks"][i] == 0 for i in opponents)
        threshold = self._call_threshold(obs, cost, opponents, terminal) if cost else 0
        realization = 1.0 if terminal else (0.93 if ip else 0.86) + (0.03 if obs["street"] == 2 else 0)
        call_equity = eq * realization
        if not terminal:
            threshold += self.config.call_margin
        if perf_counter_ns() - start > self.config.budget_ns:
            self.budget_fallbacks += 1
            return call(obs) if cost == 0 or call_equity >= threshold else passive(obs)
        if not obs["can_raise"]:
            return call(obs) if cost == 0 or call_equity >= threshold else passive(obs)
        any_allin = any(obs["stacks"][i] == 0 for i in opponents)
        own = obs["street_committed"][h]
        pot = max(obs["big_blind"], obs["pot"])
        suit_counts = [0, 0, 0, 0]
        for c in board:
            suit_counts[c % 4] += 1
        wet = max(suit_counts) >= 2 and obs["street"] <= 2
        connected = any(abs(a // 4 - b // 4) <= 2 for j, a in enumerate(board) for b in board[j + 1:])
        wet = wet or connected
        selected_eq = self.estimator.predict(features, n, rf, min(1, pressure + (0.22 if cost else 0.28)))
        value_cut = 1 / (n + 1) + self.config.value_margin + (0.10 if cost else 0)
        if (selected_eq > value_cut and (not cost or call_equity >= threshold)
                and eq > (0.64 if cost else 0.55) ** (1 + 0.35 * (n - 1))):
            if cost:
                # A raise must earn value against a materially stronger range.
                if selected_eq > max(0.58 if n == 1 else 0.43, value_cut):
                    target = own + cost + (pot + cost) * (0.70 if wet else 0.55)
                else:
                    return call(obs) if call_equity >= threshold else passive(obs)
            else:
                fraction = 0.70 if wet else 0.50
                if selected_eq < 0.64 and n == 1:
                    fraction = 0.36
                if selected_eq > 0.88:
                    fraction = 0.90 if obs["street"] == 3 else 0.75
                target = own + fraction * pot
            if target - own >= 0.63 * stack:
                target = obs["max_raise_to"]
            if not self._raise_is_safe(obs, target, opponents, selected_eq):
                return call(obs) if cost == 0 or call_equity >= threshold else passive(obs)
            return raise_to(obs, target)
        if cost:
            return call(obs) if call_equity >= threshold else passive(obs)
        # Selective bluffs in small fields. Never assume all-in players can fold.
        if n <= 2 and not any_allin and eq < 0.52 and self.config.bluff_scale > 0:
            fold_prob = prod(clamp(self.stats[i].fold_rate() + (0.10 if ip else 0)
                                  - 0.25 * pressure, 0.12, 0.85) for i in opponents)
            last_pf = raisers[-1] if raisers else None
            pfa = last_pf == h
            ranks = [c // 4 + 2 for c in hole]
            nut_blocker = any(c // 4 == 12 and suit_counts[c % 4] >= 3 for c in hole)
            draw = obs["street"] < 3 and (eq > 0.26 or any(suit_counts[c % 4] == 2
                         and hole[0] % 4 == hole[1] % 4 for c in hole))
            candidate = draw or nut_blocker or (pfa and not wet) or (ip and eq < 0.22)
            fraction = 0.38 if obs["street"] < 3 else 0.60
            frequency = (0.42 if pfa or draw else 0.24) * self.config.bluff_scale
            if (candidate and fold_prob > fraction / (1 + fraction) + 0.025
                    and self.rng.random() < frequency):
                target = own + fraction * pot
                if target - own < 0.45 * stack:
                    return raise_to(obs, target)
        return call(obs)

    @staticmethod
    def _unbeatable_board(board):
        if len(board) != 5:
            return False
        ranks = sorted(c // 4 for c in board)
        suits = [sum(c % 4 == s for c in board) for s in range(4)]
        if ranks == [8, 9, 10, 11, 12] and (max(suits) <= 2 or max(suits) == 5):
            return True
        # Quads with the best available kicker: nobody can improve the board.
        if ranks[0] == ranks[3] and ranks[4] == 12:
            return True
        return ranks[1] == ranks[4] == 12 and ranks[0] == 11
