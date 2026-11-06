"""Independent tournament policy: equity, position, ranges and bounded mixing.

This is an empirically evaluated policy, not an equilibrium solution of NLHE.
The sole solved subgame is explicitly the heads-up shove/fold abstraction.
"""
from __future__ import annotations
from math import prod
from time import perf_counter_ns

from .equity import feature_vector, has_absolute_nuts
from .icm import call_threshold
from .interface import legalize, passive
from .model import MODEL
from .preflop import TABLES, class_id
from .native import NativeEquity

NATIVE = NativeEquity(TABLES.percentile)
if MODEL.loaded and NATIVE.available:
    NATIVE.configure_model(MODEL.models, MODEL.feature_count)

DEFAULTS = dict(open_scale=1.15, call_margin=.025, value_threshold=.64,
                bluff_scale=1., shove_depth=7., icm=True, adaptation=True,
                model=True, nash=False, native=True, native_samples=256, native_mix=1.,
                budget_ns=650_000)


class Agent:
    """One table-local offline bot. `decide` uses the documented LOCAL schema.

    Instantiate afresh or call reset() at every new table. Anonymized seats
    are used only for within-table, heavily shrunk public-action statistics.
    """
    def __init__(self, seed=140014, config=None):
        self.seed = int(seed)
        self.config = dict(DEFAULTS)
        if config:
            unknown = set(config) - set(DEFAULTS)
            if unknown:
                raise ValueError(f"unknown bot parameters: {unknown}")
            self.config.update(config)
        self.reset()

    def reset(self):
        self._rng = self.seed & ((1 << 64) - 1) or 140014
        self._hand = None
        self._seen = 0
        self._opportunities = set()
        self._voluntary = set()
        self._stats = [[0., 0., 0., 0.] for _ in range(8)]
        self._feature_key = None
        self._features = None
        self._eq_cache = {}
        self.diagnostics = dict(decisions=0, budget_fallbacks=0,
                                preflop_asset_loaded=TABLES.loaded,
                                equity_asset_loaded=MODEL.loaded,
                                pushfold_asset_loaded=TABLES.pushfold is not None,
                                native_available=NATIVE.available,
                                equity_native_calls=0, equity_native_fallbacks=0,
                                equity_model_native_calls=0)

    def _random(self):
        self._rng = (self._rng + 0x9e3779b97f4a7c15) & 0xffffffffffffffff
        z = self._rng
        z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & 0xffffffffffffffff
        z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & 0xffffffffffffffff
        return ((z ^ (z >> 31)) >> 11) * (1. / (1 << 53))

    def _observe(self, obs):
        hid = obs["hand_id"]
        if self._hand is not None and hid < self._hand:
            self.reset()
        if hid != self._hand:
            self._hand, self._seen = hid, 0
            self._opportunities.clear()
            self._voluntary.clear()
        history = obs.get("history", ())
        for event in history[self._seen:]:
            seat, action = event["seat"], event["action"]
            if event["street"] == "preflop" and action in ("fold", "check", "call", "raise"):
                if seat not in self._opportunities:
                    self._stats[seat][0] += 1
                    self._opportunities.add(seat)
                if action in ("call", "raise") and seat not in self._voluntary:
                    self._stats[seat][1] += 1
                    self._voluntary.add(seat)
            if action in ("fold", "call", "raise"):
                self._stats[seat][2] += 1
                self._stats[seat][3] += action == "raise"
        self._seen = len(history)

    def _looseness(self, seat):
        if not self.config["adaptation"]:
            return .27
        seen, vpip, _, _ = self._stats[seat]
        return (7.56 + vpip) / (28. + seen)

    @staticmethod
    def _live(obs):
        return [i for i, p in enumerate(obs["players"])
                if not p["folded"] and p["stack"] + p["invested"] > 0]

    @staticmethod
    def _in_position(obs, others):
        n, button, hero = len(obs["players"]), obs["button"], obs["hero"]
        order = lambda seat: (seat - button - 1) % n
        return all(order(hero) >= order(i) for i in others if obs["players"][i]["stack"] > 0)

    @staticmethod
    def _eligible_pot(obs, cost):
        hero = obs["players"][obs["hero"]]
        cap = hero["invested"] + cost
        return sum(min(cap, p["invested"]) for p in obs["players"])

    def _opening_fraction(self, obs, seat):
        players, n = obs["players"], len(obs["players"])
        alive = [i for i, p in enumerate(players) if p.get("starting_stack", p["stack"] + p["invested"]) > 0]
        if len(alive) == 2:
            q = .78
        elif seat == obs.get("small_blind_seat"):
            q = .50
        else:
            bb = obs["big_blind_seat"]
            to_bb = (bb - seat) % n
            behind = sum(0 < (i - seat) % n <= to_bb for i in alive if i != seat)
            q = max(.15, min(.47, .50 - .053 * behind))
        return max(.08, min(.95, q * (.70 + self._looseness(seat) / .90)))

    def _ranges(self, obs, live):
        """Explicit synthetic range estimates from public actions only."""
        q = {i: 1. for i in live}
        m = {i: 1. for i in live}
        raises = 0
        street_raises = {}
        last_pf = last_post = None
        running_pot = 0
        for e in obs.get("history", ()):
            i, action, street = e["seat"], e["action"], e["street"]
            before = running_pot
            running_pot += e.get("paid", 0)
            if street == "preflop":
                if action == "raise":
                    raises += 1
                    last_pf = i
                    if i in q:
                        q[i] = max(.018, self._opening_fraction(obs, i) * (.29 ** (raises - 1)))
                        p = obs["players"][i]
                        original = p.get("starting_stack", p["stack"] + p["invested"])
                        target = e.get("to", e["amount"])
                        depth = target / obs["big_blind"]
                        if target >= original and raises == 1 and depth <= 14:
                            q[i] = min(.95, q[i] * (1.35 + max(0, 9 - depth) * .14))
                elif action == "call" and i in q:
                    q[i] = min(q[i], (.55 if not raises else .34 if raises == 1 else .095) * (.7 + self._looseness(i) / .9))
            elif i in m and action in ("call", "raise"):
                count = street_raises.get(street, 0)
                if action == "raise":
                    fraction = e.get("paid", 0) / max(obs["big_blind"], before)
                    factor = max(.19, .77 - .27 * min(2., fraction))
                    if street == "turn":
                        factor *= .86
                    elif street == "river":
                        factor *= .70
                    if count:
                        factor *= .59
                    m[i] *= factor
                    street_raises[street] = count + 1
                    last_post = i
                else:
                    m[i] *= .87 if street == "flop" else .78 if street == "turn" else .67
        return q, {i: max(.06, x) for i, x in m.items()}, raises, last_pf, last_post

    def _equity(self, features, q, m, n):
        key = (round(q, 3), round(m, 3), n)
        if key not in self._eq_cache:
            value = None
            model_value = None
            if self.config["native"] and NATIVE.available and self.config["model"]:
                model_value = NATIVE.predict_model(features, q, m, n)
                if model_value is not None:
                    self.diagnostics["equity_model_native_calls"] += 1
            exact_river = features[0] == 5 and n == 1
            mix = self.config["native_mix"]
            if model_value is not None and mix >= .999 and not exact_river:
                value = model_value
            elif self.config["native"] and NATIVE.available and n <= 3:
                hole, board = self._feature_key
                seed = 140014
                for c in tuple(hole) + tuple(board):
                    seed = (seed * 53 + c + 1) & 0xffffffffffffffff
                seed ^= round(q * 1000) * 7919 + round(m * 1000) * 104729
                self.diagnostics["equity_native_calls"] += 1
                value = NATIVE.estimate(hole, board, opponents=n, q=q, m=m,
                                        seed=seed, samples=self.config["native_samples"])
                if value is None:
                    self.diagnostics["equity_native_fallbacks"] += 1
                elif model_value is not None and not exact_river:
                    value = value * (1 - mix) + model_value * mix
            if value is None and model_value is not None:
                value = model_value
            if value is None:
                value = MODEL.predict(features, q, m, n, force_heuristic=not self.config["model"])
            self._eq_cache[key] = value
        return self._eq_cache[key]

    def decide(self, observation):
        start = perf_counter_ns()
        self.diagnostics["decisions"] += 1
        self._observe(observation)
        hero = observation["hero"]
        p = observation["players"][hero]
        if p["stack"] <= 0 or p["folded"]:
            return passive(observation)
        live = self._live(observation)
        others = [i for i in live if i != hero]
        if not others:
            return passive(observation)
        ranges = self._ranges(observation, live)
        if observation["street"] == "preflop":
            action = self._preflop(observation, others, ranges)
        else:
            action = self._postflop(observation, others, ranges, start)
        return legalize(observation, action)

    act = decide
    __call__ = decide

    def _preflop(self, obs, others, ranges):
        hero, bb = obs["hero"], obs["big_blind"]
        p, players = obs["players"][hero], obs["players"]
        stack, commit = p["stack"], p["committed"]
        cost = min(stack, obs["to_call"], max(0, max(players[i]["stack"] + players[i]["invested"] for i in others) - p["invested"]))
        maximum = stack + commit
        hand = class_id(*obs["hole_cards"])
        rank = TABLES.percentile[hand]
        q, _, raises, raiser, _ = ranges
        other_max = max(players[i]["stack"] + players[i]["committed"] for i in others)
        effective = min(maximum, other_max) / bb
        n_start = sum(x.get("starting_stack", x["stack"] + x["invested"]) > 0 for x in players)
        # In heads up, ICM points are affine in chips for any fixed payouts.
        if n_start == 2 and self.config["nash"] and effective <= 10.5:
            if not raises and hero == obs["small_blind_seat"] and obs["can_raise"]:
                probability = TABLES.nash_probability(hand, effective, "sb")
                if probability is not None:
                    return {"action": "raise", "amount": maximum} if self._random() < probability else passive(obs)
            if raises == 1 and hero == obs["big_blind_seat"] and raiser == obs["small_blind_seat"] and players[raiser]["all_in"]:
                probability = TABLES.nash_probability(hand, effective, "bb")
                if probability is not None:
                    return {"action": "call"} if self._random() < probability else passive(obs)
        if not raises:
            if all(players[i]["all_in"] for i in others):
                pot = self._eligible_pot(obs, cost)
                threshold = cost / max(1, pot + cost)
                if self.config["icm"] and len(others) == 1:
                    threshold = call_threshold(obs, others[0], cost, pot)
                return {"action": "call"} if TABLES.equity(hand, 1., len(others)) >= threshold else passive(obs)
            limpers = sum(e["action"] == "call" and e["street"] == "preflop" for e in obs.get("history", ()))
            if effective <= self.config["shove_depth"] and obs["can_raise"]:
                ev = self._shove_ev(obs, others, hand)
                if ev > .035 * bb:
                    return {"action": "raise", "amount": maximum}
                if cost and cost <= .14 * stack:
                    odds = cost / max(1, self._eligible_pot(obs, cost) + cost)
                    if TABLES.equity(hand, .7, len(others)) > odds + .11:
                        return {"action": "call"}
                return passive(obs)
            opening = self._opening_fraction(obs, hero) * self.config["open_scale"]
            if limpers:
                opening = min(opening, .26 / (1 + .15 * (limpers - 1)))
                if hero == obs["big_blind_seat"]:
                    opening = min(opening, .18)
            if rank < opening and obs["can_raise"]:
                size = bb * ((2.25 if effective <= 35 else 2.6) + .90 * limpers)
                return {"action": "raise", "amount": round(size)}
            hi, lo = max(c >> 2 for c in obs["hole_cards"]), min(c >> 2 for c in obs["hole_cards"])
            suited = obs["hole_cards"][0] & 3 == obs["hole_cards"][1] & 3
            if limpers and effective >= 22 and cost <= bb and (hi == lo or suited and hi - lo <= 2 and lo >= 3) and rank < .50:
                return {"action": "call"}
            return passive(obs)
        rq = q.get(raiser, .18)
        contesting = [i for i in others if players[i]["committed"] >= obs["current_bet"] or players[i]["all_in"]]
        n = max(1, len(contesting))
        eq = TABLES.equity(hand, rq, n)
        pot = self._eligible_pot(obs, cost)
        threshold = cost / max(1, pot + cost)
        allin = cost >= stack or not any(players[i]["stack"] for i in others)
        if self.config["icm"] and len(others) == 1 and (allin or cost >= .25 * stack):
            threshold = call_threshold(obs, raiser, cost, pot)
        ip = self._in_position(obs, others)
        if allin:
            return {"action": "call"} if eq >= threshold + .005 else passive(obs)
        value_cutoff = .61 if raises == 1 else .57
        if eq > value_cutoff and rank < (.09 if rq >= .28 else .045 if raises == 1 else .022) and obs["can_raise"]:
            size = obs["current_bet"] * (3.05 if ip else 3.75)
            size += sum(max(0, players[i]["committed"] - bb) for i in others if i != raiser)
            if size >= maximum * .35 or effective < 20:
                size = maximum
            return {"action": "raise", "amount": round(size)}
        h0, h1 = obs["hole_cards"]
        hi, lo = max(h0 >> 2, h1 >> 2), min(h0 >> 2, h1 >> 2)
        suited = h0 & 3 == h1 & 3
        if raises == 1 and rq > .27 and effective >= 35 and hi == 12 and lo <= 4 and suited and obs["can_raise"] and self._random() < .18 * self.config["bluff_scale"]:
            return {"action": "raise", "amount": round(obs["current_bet"] * (3.05 if ip else 3.75))}
        realization = (.86 if ip else .77) + .055 * suited + .055 * (hi == lo)
        if raises > 1:
            realization -= .02
        risk_fraction = cost / max(1, stack)
        margin = self.config["call_margin"] + .025 * min(1., risk_fraction * 2)
        if eq * min(.98, realization) >= threshold + margin:
            return {"action": "call"}
        return passive(obs)

    def _shove_ev(self, obs, others, hand):
        """Approximate open-shove EV with blocker-aware call frequencies.

        Integrates no-call/exactly-one-call outcomes and conservatively models
        multiple callers. This is a policy approximation, not a solved 8p game.
        """
        hero, players, bb = obs["hero"], obs["players"], obs["big_blind"]
        p = players[hero]
        maximum = p["stack"] + p["committed"]
        probabilities, values = [], []
        # One exact representative ICM premium bounds the expensive branch.
        # Interpolate smaller risks from the largest covering opponent instead
        # of solving seven separate three-outcome ICM calculations per action.
        premium = 0.
        ref_risk = 1.
        if self.config["icm"] and obs.get("payouts"):
            ref = max(others, key=lambda i: players[i]["stack"] + players[i]["committed"])
            op = players[ref]
            ref_risk = min(p["stack"], max(0, op["stack"] + op["committed"] - p["committed"]))
            ref_add = max(0, min(maximum, op["stack"] + op["committed"]) - op["committed"])
            ref_pot = self._eligible_pot(obs, ref_risk)
            if ref_risk >= .25 * p["stack"]:
                t = call_threshold(obs, ref, ref_risk, ref_pot + ref_add, villain_add=ref_add)
                premium = max(0., t - ref_risk / max(1, ref_pot + ref_add + ref_risk))
        for i in others:
            other = players[i]
            depth = min(maximum, other["stack"] + other["committed"]) / bb
            fraction = min(.95, max(.045, 1.8 / (depth + 1.5)))
            # Earlier callers face potential overcalls behind them.
            remaining = sum(0 < (j - i) % len(players) < (hero - i) % len(players) for j in others if j != i)
            fraction *= max(.57, 1 - .065 * remaining)
            fraction *= .60 + self._looseness(i) / .675
            fraction = min(1., max(.025, fraction))
            already_in = other["committed"] >= maximum or other["all_in"]
            probability = 1. if already_in else TABLES.mass(hand, fraction)
            risk = min(p["stack"], max(0, other["stack"] + other["committed"] - p["committed"]))
            add = max(0, min(maximum, other["stack"] + other["committed"]) - other["committed"])
            pot = self._eligible_pot(obs, risk)
            eq = TABLES.equity(hand, fraction)
            risk_premium = premium * min(1., risk / max(1, ref_risk)) * (pot + add + risk)
            values.append(eq * (pot + risk + add) - risk - risk_premium)
            probabilities.append(probability)
        no_call = prod(1 - p for p in probabilities)
        ev = no_call * obs["pot"]
        single_mass = 0.
        for k, probability in enumerate(probabilities):
            mass = probability * prod(1 - x for j, x in enumerate(probabilities) if j != k)
            single_mass += mass
            ev += mass * values[k]
        multiple = max(0., 1 - no_call - single_mass)
        if multiple:
            # Costly overcalls are deliberately valued below a heads-up call.
            ev += multiple * (min(values) - .22 * p["stack"])
        return ev

    def _postflop(self, obs, others, ranges, start):
        hero, players, bb = obs["hero"], obs["players"], obs["big_blind"]
        p = players[hero]
        cost = min(p["stack"], obs["to_call"], max(0, max(players[i]["stack"] + players[i]["invested"] for i in others) - p["invested"]))
        stack = p["stack"]
        key = (tuple(obs["hole_cards"]), tuple(obs["board"]))
        if key != self._feature_key:
            self._features = feature_vector(*key)
            self._feature_key = key
            self._eq_cache.clear()
        f = self._features
        q, m, _, last_pf, aggressor = ranges
        n = len(others)
        # Geometric means preserve the overall tightening of heterogeneous
        # ranges while an explicit penalty protects multiway marginal calls.
        qbar = prod(q[i] for i in others) ** (1 / n)
        mbar = prod(m[i] for i in others) ** (1 / n)
        if cost and aggressor in others:
            qbar = min(qbar, q[aggressor])
            mbar = min(mbar, m[aggressor])
        if len(obs["board"]) == 5 and n == 1 and self.config["native"] and NATIVE.available:
            cutoffs = (mbar, max(.06, mbar * .64))
            exact = NATIVE.river_equities(obs["hole_cards"], obs["board"], q=qbar, m_values=cutoffs)
            self.diagnostics["equity_native_calls"] += 1
            if exact is None:
                self.diagnostics["equity_native_fallbacks"] += 1
            else:
                for cutoff, value in zip(cutoffs, exact):
                    self._eq_cache[(round(qbar, 3), round(cutoff, 3), 1)] = value
        eq = self._equity(f, qbar, mbar, n)
        if cost:
            cap = p["invested"] + cost
            short = [i for i in others if players[i]["all_in"] and players[i]["invested"] < cap]
            if short:
                boundary = max(players[i]["invested"] for i in short)
                deep = [i for i in others if players[i]["invested"] > boundary or
                        not players[i]["all_in"] and players[i]["stack"] + players[i]["invested"] > boundary]
                final_pot = self._eligible_pot(obs, cost) + cost
                main = sum(min(boundary, x["invested"] + (cost if j == hero else 0)) for j, x in enumerate(players))
                if deep and len(deep) < n and main < final_pot * .90:
                    dq = prod(q[i] for i in deep) ** (1 / len(deep))
                    dm = prod(m[i] for i in deep) ** (1 / len(deep))
                    de = self._equity(f, dq, dm, len(deep))
                    # One additional prediction repairs the dominant side pot.
                    # Several differently capped short stacks share a lower-pot
                    # approximation; this does not alter engine pot settlement.
                    eq = (eq * main + de * (final_pot - main)) / final_pot
        ip = self._in_position(obs, others)
        river = len(obs["board"]) == 5
        pot = self._eligible_pot(obs, cost)
        threshold = cost / max(1, pot + cost)
        if cost and n == 1 and self.config["icm"] and (cost >= .20 * stack or players[others[0]]["all_in"]):
            threshold = call_threshold(obs, others[0], cost, pot)
        # Proven nuts cannot be beaten; avoid inference approximation folding
        # such hands. Board-only ties still obey pot odds and side-pot limits.
        nuts = has_absolute_nuts(obs["hole_cards"], obs["board"])
        if nuts:
            eq = max(eq, 1. / (n + 1))
            if f[1] >= 7 and (f[33] < f[1] or f[2] > f[34]):
                eq = .999
        allin = cost >= stack or not any(players[i]["stack"] for i in others)
        can_continue = eq >= threshold + (.004 if allin or river else self.config["call_margin"])
        if not river and not allin:
            # Equity is all-in showdown share. Out-of-position weak hands
            # realize less of it when future bets can force a fold.
            realization = 1. if f[1] >= 2 or f[21] == 4 or f[28] >= 2 else .94 if ip else .86
            can_continue = eq * realization >= threshold + self.config["call_margin"] + .01 * max(0, n - 1)
        if nuts:
            # With an unbeatable hand, every matching contribution either
            # increases our split or folds; a call cannot lose contestable chips.
            can_continue = True
        if cost and not can_continue:
            return passive(obs)
        if not obs["can_raise"] or allin:
            return {"action": "call"} if cost else {"action": "check"}
        if perf_counter_ns() - start > self.config["budget_ns"]:
            self.diagnostics["budget_fallbacks"] += 1
            return {"action": "call"} if cost else {"action": "check"}
        # Value bets are checked against a stronger continuing range, rather
        # than just equity against hands that would mostly fold.
        continue_m = max(.06, mbar * (.64 if river else .73))
        e_continue = self._equity(f, qbar, continue_m, n)
        threshold_value = self.config["value_threshold"] + .035 * max(0, n - 1)
        if cost:
            threshold_value += .045
        if nuts and f[33] < f[1]:
            e_continue = max(e_continue, .98)
        if e_continue >= threshold_value:
            fraction = .68 if e_continue >= .82 or n > 1 else .46
            if river and e_continue >= .91:
                fraction = .95
            if not river and f[20] >= 2 and f[26] >= 4:
                fraction = max(fraction, .70)
            target = p["committed"] + cost + round((pot + cost) * fraction)
            if cost:
                target = max(target, obs["current_bet"] * 2.65)
            maximum = p["committed"] + stack
            if maximum - target <= .23 * (pot + max(0, target - p["committed"])):
                target = maximum
            return {"action": "raise", "amount": round(target)}
        # Controlled low-frequency semi-bluffs and positional probes. Against
        # multiple live hands, bluff opportunities contract sharply.
        if not cost and n <= 2:
            draw = not river and (f[21] == 4 and f[22] > 0 or f[28] >= 2)
            dry = f[20] <= 2 and f[27] <= 3
            blocker = f[23] == 14 and f[39] >= 2
            weak = eq < .41 and f[1] <= 1
            frequency = (.20 if draw else .15 if weak and dry and last_pf == hero else .09 if weak and ip else .06 if weak and river and blocker else 0.)
            frequency *= self.config["bluff_scale"] * (1 if n == 1 else .40)
            if self._random() < frequency:
                fraction = .40 if not river else .65
                return {"action": "raise", "amount": p["committed"] + round(max(bb, pot * fraction))}
        return {"action": "call"} if cost else {"action": "check"}
