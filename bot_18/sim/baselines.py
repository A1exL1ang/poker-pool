"""Independent deliberately simple opponents for local robustness checks.

These are heuristic opponents, not verified equilibrium strategies. They are
diverse enough to reveal obvious errors but cannot establish contest strength.
No baseline calls the candidate's policy or learned equity model.
"""
from collections import Counter
import random
from equity.evaluator import evaluate

STYLES = ("calling", "nit", "tag", "lag", "maniac", "random", "pushfold",
          "range", "trap", "pressure")


def preflop_strength(hole):
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    if a == b:
        return 0.58 + (a - 2) * 0.0335
    suited = hole[0] % 4 == hole[1] % 4
    connected = max(0, 4 - (a - b))
    return min(0.92, 0.10 + a * 0.025 + b * 0.019 + 0.055 * suited
               + 0.016 * connected + 0.06 * (a == 14))


def postflop_strength(hole, board):
    """Coarse showdown/draw proxy intentionally independent of learned model."""
    value = evaluate(hole + board)
    category = value >> 20
    hero_ranks = [c // 4 + 2 for c in hole]
    board_ranks = [c // 4 + 2 for c in board]
    counts = Counter(board_ranks)
    if category >= 4:
        strength = (0.93, 0.96, 0.985, 0.997, 0.999)[category - 4]
        if len(board) == 5 and evaluate(board) == value:
            strength = 0.42
    elif category == 3:
        strength = 0.91 if any(counts.get(r, 0) in (1, 2) for r in hero_ranks) else 0.42
    elif category == 2:
        if hero_ranks[0] == hero_ranks[1] or len(set(hero_ranks) & set(board_ranks)) == 2:
            strength = 0.83
        elif any(r in counts for r in hero_ranks):
            strength = 0.68
        else:
            strength = 0.35
    elif category == 1:
        if hero_ranks[0] == hero_ranks[1]:
            pair = hero_ranks[0]
            strength = 0.79 if pair > max(board_ranks) else 0.59 if pair > sorted(board_ranks)[len(board_ranks)//2] else 0.43
        elif any(r in counts for r in hero_ranks):
            pair = max(r for r in hero_ranks if r in counts)
            strength = 0.72 if pair == max(board_ranks) else 0.51 if pair >= sorted(set(board_ranks))[-2] else 0.38
            strength += (max(hero_ranks) - 8) * 0.008
        else:
            strength = 0.22 + (max(hero_ranks) - 8) * 0.02
    else:
        strength = 0.10 + 0.023 * (max(hero_ranks) - 2)
    if len(board) < 5:
        suit_counts = Counter(c % 4 for c in hole + board)
        flush_draw = any(n == 4 and any(c % 4 == suit for c in hole) for suit, n in suit_counts.items())
        ranks = set(hero_ranks + board_ranks)
        if 14 in ranks:
            ranks.add(1)
        straight_draw = max(len(ranks & set(range(low, low + 5))) for low in range(1, 11)) == 4
        strength += 0.12 * flush_draw + 0.07 * straight_draw
    return min(0.999, max(0.01, strength))


class Baseline:
    def __init__(self, style="tag", seed=0):
        if style not in STYLES:
            raise ValueError(f"unknown baseline {style}")
        self.style, self.seed = style, seed
        self.rng = random.Random(seed)
        self.skilled = _RangeOpponent(style, seed) if style in ("range", "trap", "pressure") else None

    def reset(self):
        self.rng.seed(self.seed)
        if self.skilled is not None:
            self.skilled.reset()

    @staticmethod
    def _passive(obs, call=True):
        return {"action": "check" if not obs["to_call"] else "call" if call else "fold"}

    def _raise(self, obs, amount):
        if not obs["can_raise"]:
            return self._passive(obs)
        amount = min(obs["max_raise_to"], max(obs["min_raise_to"], int(amount)))
        return {"action": "raise", "amount": amount}

    def act(self, obs):
        if self.skilled is not None:
            return self.skilled.act(obs)
        hero = obs["players"][obs["hero"]]
        call = obs["to_call"]
        pot = obs["pot"]
        bb = obs["big_blind"]
        strength = preflop_strength(obs["hole"]) if not obs["board"] else postflop_strength(obs["hole"], obs["board"])
        price = call / max(1, pot + call)
        roll = self.rng.random()
        if self.style == "calling":
            if strength > 0.95 and roll < 0.6:
                return self._raise(obs, hero["bet"] + call + pot * 0.7)
            return self._passive(obs, not obs["board"] or strength + 0.20 > price)
        if self.style == "random":
            if roll < 0.22:
                return self._raise(obs, hero["bet"] + call + max(bb, pot * self.rng.choice([0.5, 1, 2])))
            return self._passive(obs, roll < 0.78)
        if self.style == "maniac":
            if roll < 0.64:
                return self._raise(obs, hero["bet"] + call + max(bb, pot * self.rng.choice([0.6, 1, 1.6])))
            return self._passive(obs, strength > max(0.2, price - 0.1))
        if self.style == "pushfold":
            if not obs["board"]:
                bb_stack = hero["stack"] / bb
                threshold = 0.49 if bb_stack < 8 else 0.60 if bb_stack < 20 else 0.80
                if strength > threshold:
                    return self._raise(obs, obs["max_raise_to"])
                return self._passive(obs, call <= bb and strength > 0.50)
            return self._passive(obs, strength > price + 0.1)
        nit = self.style == "nit"
        lag = self.style == "lag"
        if not obs["board"]:
            raises = sum(h["kind"] == "raise" for h in obs["history"] if h["street"] == "preflop")
            remaining = sum(not p["folded"] for p in obs["players"])
            threshold = (0.71 if nit else 0.56 if lag else 0.64) - max(0, 5 - remaining) * 0.025
            if raises:
                threshold += min(0.16, raises * 0.05)
            if strength > threshold + 0.08 or (raises == 0 and strength > threshold):
                size = bb * (3 + 2 * raises) if raises == 0 else max(bb * 6, (hero["bet"] + call) * 3)
                return self._raise(obs, size)
            willing = strength > threshold - (0.03 if nit else 0.09) and call < min(hero["stack"] * 0.2, bb * 6)
            if hero["stack"] <= call:
                willing = strength > max(threshold, price + 0.22)
            return self._passive(obs, willing)
        threshold = 0.82 if nit else 0.67 if lag else 0.75
        bluff = 0.015 if nit else 0.17 if lag else 0.06
        if strength > threshold or (call == 0 and roll < bluff):
            return self._raise(obs, hero["bet"] + call + max(bb, pot * (0.75 if nit else 0.60)))
        premium = 0.20 if nit else 0.05 if lag else 0.12
        return self._passive(obs, strength > price + premium)


class _RangeOpponent:
    """Frozen, independent stronger heuristics reserved for held-out evaluation.

    Shares the offline *equity estimator* as explicitly permitted, but imports
    no candidate policy, configuration, opponent statistics or decision logic.
    Range, trap, and pressure have distinct action selection. These are not
    solved strategies; the model sharing is an evaluation limitation.
    """
    def __init__(self, style, seed):
        self.style, self.seed = style, seed
        self.rng = random.Random(seed)

    def reset(self):
        self.rng.seed(self.seed)

    def act(self, obs):
        from equity.model import estimate, estimate_premium, forced_tie
        from equity.preflop import preflop_percentile

        hero = obs["players"][obs["hero"]]
        villains = [p for p in obs["players"] if not p["folded"] and p["seat"] != obs["hero"]]
        n = len(villains)
        owed, pot, blind = obs["to_call"], obs["pot"], obs["big_blind"]
        passive = {"action": "call" if owed else "check"}
        fold = {"action": "fold" if owed else "check"}
        cap = hero["invested"] + owed
        contest = sum(min(cap, p["invested"]) for p in obs["players"])
        required = owed / max(1, contest + owed)
        raise_events = [a for a in obs["history"] if a["street"] == "preflop" and a["kind"] == "raise"]
        total_wager = max(p["bet"] for p in obs["players"])
        later = sum((p["seat"] - obs["button"] - 1) % 8 > (obs["hero"] - obs["button"] - 1) % 8 for p in villains)
        last_to_act = later == 0
        depth = min(hero["stack"] + hero["bet"], max((p["stack"] + p["bet"] for p in villains), default=0)) / blind
        percentile = preflop_percentile(obs["hole"])
        aggressive = self.style == "pressure"

        def wager(target):
            if not obs["can_raise"]:
                return passive
            return {"action": "raise", "amount": min(obs["max_raise_to"], max(obs["min_raise_to"], int(target)))}

        if not obs["board"]:
            if not raise_events:
                limit = (.64, .44, .31, .25, .21, .18, .16, .14)[min(n, 7)]
                if obs["hero"] == obs["button"]:
                    limit = .73 if n == 1 else .48
                limit += .07 if aggressive else -.02 if self.style == "trap" else 0
                if percentile < limit:
                    if depth <= 9:
                        return wager(obs["max_raise_to"])
                    if self.style == "trap" and percentile < .035 and self.rng.random() < .32:
                        return passive
                    limps = sum(a["kind"] == "call" for a in obs["history"] if a["street"] == "preflop")
                    return wager(blind * (2.4 + .9 * limps))
                suited = obs["hole"][0] % 4 == obs["hole"][1] % 4
                if owed <= blind / 2 and suited and percentile < .65:
                    return passive
                return fold
            count = len(raise_events)
            opening = .27 if len(villains) <= 2 else .17
            assumed_range = opening if count == 1 else .055 if count == 2 else .025
            if aggressive:
                assumed_range *= 1.25
            equity = estimate(obs["hole"], [], assumed_range, max(1, n))
            if total_wager >= 18 * blind and n == 1:
                equity = min(equity, estimate_premium(obs["hole"], nutted=count >= 3))
            if owed >= hero["stack"] or not any(p["stack"] for p in villains):
                return passive if equity > required + .035 else fold
            if percentile < (.045 if count == 1 else .016) and equity > .50:
                if self.style == "trap" and self.rng.random() < .36:
                    return passive
                size = total_wager * (3.2 + .7 * max(0, n - 1))
                return wager(obs["max_raise_to"] if depth < 22 else size)
            position_discount = .03 if last_to_act else .08
            return passive if equity > required + position_discount and percentile < assumed_range * 1.7 else fold

        if forced_tie(obs["hole"], obs["board"]):
            return passive
        prior_range = .20 if raise_events else .48
        if len(raise_events) > 1:
            prior_range = .07
        equity = estimate(obs["hole"], obs["board"], prior_range, max(1, n))
        street_bets = [a for a in obs["history"] if a["street"] == obs["street"] and a["kind"] == "raise"]
        was_raiser = bool(raise_events and raise_events[-1]["seat"] == obs["hero"])
        rank_mask = {c // 4 for c in obs["hole"] + obs["board"]}
        if 12 in rank_mask:
            rank_mask.add(-1)
        run_draw = max(len(rank_mask & set(range(i, i + 5))) for i in range(-1, 9)) == 4
        suits = Counter(c % 4 for c in obs["hole"] + obs["board"])
        flush_draw = any(v == 4 for v in suits.values())
        draw = len(obs["board"]) < 5 and (run_draw or flush_draw)
        terminal = len(obs["board"]) == 5 or owed >= hero["stack"] or not any(p["stack"] for p in villains)
        if owed:
            # Independent odds update: a wager favors the opponent's winning
            # combinations. Larger bets/re-raises get a larger likelihood.
            likelihood = 1.25 + .7 * min(3, owed / max(1, pot - owed)) + .7 * max(0, len(street_bets) - 1)
            if len(obs["board"]) == 5:
                likelihood += .65
            adjusted = equity / max(1e-9, equity + (1 - equity) * likelihood)
            buffer = .015 if terminal else .04
            if adjusted < required + buffer:
                return fold
            raise_cut = .79 if self.style == "trap" else .87
            if adjusted > raise_cut and obs["can_raise"]:
                return wager(hero["bet"] + owed + (pot + owed) * (.95 if aggressive else .7))
            if aggressive and draw and n == 1 and adjusted > .47 and self.rng.random() < .18:
                return wager(hero["bet"] + owed + .9 * (pot + owed))
            return passive
        value_cut = .65 if len(obs["board"]) == 3 else .70 if len(obs["board"]) == 4 else .75
        value_cut += .03 * max(0, n - 1)
        bet_value = equity > value_cut
        if self.style == "trap" and bet_value and len(obs["board"]) < 5 and self.rng.random() < .48:
            return passive
        bluff_chance = .04
        if n == 1 and len(obs["board"]) == 3 and was_raiser:
            bluff_chance = .38 if aggressive else .19
        elif n == 1 and draw:
            bluff_chance = .30 if aggressive else .13
        elif n == 1 and last_to_act and aggressive:
            bluff_chance = .17
        if bet_value or (equity < .55 and self.rng.random() < bluff_chance):
            fraction = .45 if len(obs["board"]) == 3 else .72
            if aggressive and (equity > .91 or equity < .3):
                fraction = 1.25
            return wager(hero["bet"] + max(blind, pot * fraction))
        return passive

