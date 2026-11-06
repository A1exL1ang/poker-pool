"""Original local comparison policies, not externally supplied poker bots.

These cover several exploitable styles and a uniform-equity reference. They
are deliberately modest opponents; beating them is not proof of contest rank.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random


def _hero(obs):
    return next(p for p in obs["players"] if p["seat"] == obs["hero"])


def passive(obs):
    return {"action": "check" if obs["to_call"] == 0 else "fold"}


def call(obs):
    return {"action": "check" if obs["to_call"] == 0 else "call"}


def raise_to(obs, target):
    if not obs["can_raise"]:
        return call(obs)
    target = min(obs["max_raise_to"], max(obs["min_raise_to"], int(target)))
    if target <= obs["current_bet"]:
        return call(obs)
    return {"action": "raise", "amount": target}


def preflop_strength(hole):
    high, low = sorted((c // 4 + 2 for c in hole), reverse=True)
    suited = hole[0] % 4 == hole[1] % 4
    if high == low:
        return 0.49 + (high - 2) * 0.033
    strength = 0.22 + 0.025 * (high - 2) + 0.015 * (low - 2)
    strength += 0.045 if suited else 0
    gap = high - low
    strength += {1: 0.04, 2: 0.015, 3: -0.005}.get(gap, -0.025)
    if high == 14 and low >= 10:
        strength += 0.045
    return min(0.9, max(0.15, strength))


def postflop_strength(hole, board):
    """Made hand plus modest draw heuristic; returns a rough HU equity proxy."""
    from lab.simulator import _score_five
    from itertools import combinations
    score = max(_score_five(c) for c in combinations(hole + board, 5))
    category = score[0]
    base = [0.18, 0.48, 0.70, 0.80, 0.87, 0.90, 0.95, 0.98, 0.995][category]
    ranks = [c // 4 + 2 for c in hole]
    board_ranks = [c // 4 + 2 for c in board]
    if category == 0:
        base += 0.055 * sum(r > max(board_ranks) for r in ranks)
    if category == 1:
        pair = score[1]
        if pair not in ranks:
            base -= 0.17  # A board pair does not give hero a strong private hand.
        else:
            base += 0.025 * (pair - max(board_ranks))
    if len(board) < 5:
        suits = Counter(c % 4 for c in hole + board)
        base += 0.13 if max(suits.values()) == 4 and any(suits[c % 4] == 4 for c in hole) else 0
        rank_set = set(ranks + board_ranks)
        if 14 in rank_set:
            rank_set.add(1)
        straight_near = max(len(rank_set.intersection(range(start, start + 5))) for start in range(1, 11))
        base += 0.08 if straight_near == 4 else 0
    return min(0.995, max(0.02, base))


class CheckFold:
    def act(self, obs):
        return passive(obs)


class CallingStation:
    def __init__(self, seed=1):
        self.rng = random.Random(seed)

    def act(self, obs):
        strength = preflop_strength(obs["hole_cards"]) if not obs["board"] else postflop_strength(obs["hole_cards"], obs["board"])
        cost = obs["to_call"]
        stack = _hero(obs)["stack"]
        if cost == 0:
            if strength > 0.89 and obs["can_raise"]:
                return raise_to(obs, obs["current_bet"] + max(obs["big_blind"], obs["pot"] * 0.45))
            return call(obs)
        odds = cost / (obs["pot"] + cost)
        if cost > 0.65 * stack and strength < (0.56 if not obs["board"] else 0.63):
            return passive(obs)
        if strength > max(0.17, odds - 0.06):
            return call(obs)
        return passive(obs)


class TightValue:
    def __init__(self, seed=1):
        self.rng = random.Random(seed)

    def act(self, obs):
        active = [p for p in obs["players"] if not p["folded"]]
        strength = preflop_strength(obs["hole_cards"]) if not obs["board"] else postflop_strength(obs["hole_cards"], obs["board"])
        cost = obs["to_call"]
        bb = obs["big_blind"]
        pot = obs["pot"]
        if not obs["board"]:
            opponents = len(active) - 1
            threshold = 0.48 + 0.018 * min(5, opponents - 1)
            raised = obs["current_bet"] > bb
            if raised:
                threshold += min(0.16, 0.035 * (obs["current_bet"] / bb - 1))
            if strength > threshold + 0.13 and obs["can_raise"]:
                return raise_to(obs, max(3 * bb, obs["current_bet"] * 3))
            if strength >= threshold or cost <= 0.02 * pot:
                return call(obs)
            return passive(obs)
        odds = cost / (pot + cost) if cost else 0
        adjusted = strength ** (0.65 + 0.35 * (len(active) - 1))
        if adjusted > 0.73 and obs["can_raise"]:
            return raise_to(obs, obs["current_bet"] + max(bb, 0.65 * (pot + cost)))
        if cost == 0 or adjusted > odds + 0.10:
            return call(obs)
        return passive(obs)


class Aggressive:
    def __init__(self, seed=1):
        self.rng = random.Random(seed)

    def act(self, obs):
        strength = preflop_strength(obs["hole_cards"]) if not obs["board"] else postflop_strength(obs["hole_cards"], obs["board"])
        cost = obs["to_call"]
        pot = obs["pot"]
        bb = obs["big_blind"]
        raised = any(e["action"] == "raise" and e["street"] == obs["street"] for e in obs["history"])
        draw = len(obs["board"]) in (3, 4) and 0.32 < strength < 0.65
        if obs["can_raise"] and (strength > (0.61 if raised else 0.45) or (draw and self.rng.random() < 0.3) or (cost <= bb and not raised and self.rng.random() < 0.19)):
            return raise_to(obs, obs["current_bet"] + max(2.3 * bb, (pot + cost) * 0.8))
        if cost == 0 or strength > cost / (pot + cost) + (0.06 if raised else -0.02):
            return call(obs)
        return passive(obs)


class RandomLegal:
    def __init__(self, seed=1):
        self.rng = random.Random(seed)

    def act(self, obs):
        x = self.rng.random()
        if x < 0.2:
            return passive(obs)
        if x < 0.82 or not obs["can_raise"]:
            return call(obs)
        fraction = self.rng.choice((0.4, 0.75, 1.25))
        return raise_to(obs, obs["current_bet"] + max(obs["big_blind"], fraction * (obs["pot"] + obs["to_call"])))


class UniformEquity:
    """Reference policy using uniform opponent hands, no action-conditioned ranges."""
    def __init__(self, seed=1, samples=96):
        self.rng = random.Random(seed)
        self.seed = seed
        self.samples = samples
        from bot13.native import NativeEvaluator
        self.evaluator = NativeEvaluator()

    def act(self, obs):
        self.seed = (self.seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        opponents = max(1, sum(not p["folded"] for p in obs["players"]) - 1)
        strength, _ = self.evaluator.equity(obs["hole_cards"], obs["board"], opponents=opponents, seed=self.seed, samples=self.samples)
        cost = obs["to_call"]
        pot = obs["pot"]
        odds = cost / (pot + cost) if cost else 0
        if obs["can_raise"] and (strength > max(0.58, 1 / (opponents + 1) + 0.25)):
            return raise_to(obs, obs["current_bet"] + max(obs["big_blind"] * 2.2, 0.65 * (pot + cost)))
        if cost == 0 or strength > odds + (0.06 if obs["board"] else 0.10):
            return call(obs)
        return passive(obs)


_PERCENTILES = None


def _preflop_percentiles():
    """Combination-weighted strength ranks from this bot's own offline asset.

    This is a laboratory stress ordering, not a claimed optimal shove chart.
    The asset and sorting are loaded during construction, outside decisions.
    """
    global _PERCENTILES
    if _PERCENTILES is not None:
        return _PERCENTILES
    ranks = "23456789TJQKA"
    entries = []
    for hi in range(12, -1, -1):
        for lo in range(hi, -1, -1):
            for suffix in ([""] if hi == lo else ["s", "o"]):
                key = ranks[hi] + ranks[lo] + suffix
                example = [4 * hi, 4 * lo + (1 if suffix == "o" or hi == lo else 0)]
                entries.append((key, preflop_strength(example), 6 if hi == lo else 4 if suffix == "s" else 12))
    path = Path(__file__).resolve().parents[1] / "assets" / "preflop.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))["hands"]
        entries = [(key, float(data[key]["random"][0]), count) for key, _, count in entries]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    entries.sort(key=lambda item: (-item[1], item[0]))
    cumulative = 0
    result = {}
    for key, _, count in entries:
        result[key] = (cumulative + count / 2) / 1326
        cumulative += count
    assert cumulative == 1326
    _PERCENTILES = result
    return result


def _hand_key(hole):
    ranks = "23456789TJQKA"
    high, low = sorted((hole[0] // 4, hole[1] // 4), reverse=True)
    return ranks[high] + ranks[low] + ("" if high == low else "s" if hole[0] % 4 == hole[1] % 4 else "o")


def _board_profile(obs):
    from bot13.cards import evaluate, features
    f = features(obs["hole_cards"], obs["board"])
    category = evaluate(obs["hole_cards"] + obs["board"]) >> 24
    private_pair = f["pair_rank"] > 0 or f["overpair"]
    private_two_pair = category >= 2 and private_pair
    strong = category >= 4 or f["pocket_set"] or private_two_pair
    draw = f["flush_draw"] or f["open_ended"]
    return f, category, private_pair, strong, draw


class ShoveWide:
    """Stress overfolds: open-jam roughly top 60%, call jams much tighter.

    This intentionally extreme original opponent also tests whether range
    inference adapts to wide jams rather than assuming every large bet is AA.
    """
    def __init__(self, seed=1):
        self.percentiles = _preflop_percentiles()

    def act(self, obs):
        if not obs["board"]:
            percentile = self.percentiles[_hand_key(obs["hole_cards"])]
            raised = obs["current_bet"] > obs["big_blind"]
            if raised:
                return call(obs) if percentile <= 0.14 else passive(obs)
            if percentile <= 0.60:
                return raise_to(obs, obs["max_raise_to"]) if obs["can_raise"] else call(obs)
            return passive(obs)
        f, category, private_pair, strong, draw = _board_profile(obs)
        cost = obs["to_call"]
        if strong and obs["can_raise"]:
            return raise_to(obs, obs["max_raise_to"])
        if cost == 0:
            return call(obs)
        odds = cost / (obs["pot"] + cost)
        if (private_pair and odds < 0.31) or (draw and len(obs["board"]) < 5 and odds < 0.24):
            return call(obs)
        return passive(obs)


class PremiumNit:
    """Stress overcalling: only QQ+/AK voluntarily enter preflop pots.

    Large deep-stack all-ins are called with KK+; postflop aggression requires
    strong private value. Free blind checks can still produce other hands.
    """
    def __init__(self, seed=1):
        pass

    def act(self, obs):
        key = _hand_key(obs["hole_cards"])
        bb = obs["big_blind"]
        if not obs["board"]:
            premium = key in {"AA", "KK", "QQ", "AKs", "AKo"}
            if not premium:
                return passive(obs)
            if obs["to_call"] >= 25 * bb:
                return call(obs) if key in {"AA", "KK"} else passive(obs)
            if obs["can_raise"]:
                target = 3 * bb if obs["current_bet"] <= bb else 3.4 * obs["current_bet"]
                return raise_to(obs, target)
            return call(obs)
        f, category, private_pair, strong, draw = _board_profile(obs)
        cost = obs["to_call"]
        pot = obs["pot"]
        if strong and obs["can_raise"]:
            return raise_to(obs, obs["current_bet"] + max(bb, 0.9 * (pot + cost)))
        if cost == 0:
            return call(obs)
        odds = cost / (pot + cost)
        good_pair = f["overpair"] or (f["top_pair"] and f["kicker"] >= 11)
        if (strong and odds < 0.48) or (good_pair and odds <= 0.27) or (draw and len(obs["board"]) < 5 and odds <= 0.18):
            return call(obs)
        return passive(obs)


class MinraisePressure:
    """Stress minimum-defense errors with positional steals and small cbets.

    Uses an original bounded mixed heuristic, private value, and draws. Its
    point is adversarial coverage, not a second independently submitted bot.
    """
    def __init__(self, seed=1):
        self.rng = random.Random(seed)
        self.percentiles = _preflop_percentiles()

    def act(self, obs):
        hero = _hero(obs)
        bb = obs["big_blind"]
        cost = obs["to_call"]
        pot = obs["pot"]
        live = [p for p in obs["players"] if not p["folded"]]
        if not obs["board"]:
            percentile = self.percentiles[_hand_key(obs["hole_cards"])]
            seat_count = obs.get("seat_count", len(obs["players"]))
            bb_seat = obs.get("big_blind_seat")
            if bb_seat is None:
                bb_seat = max(live, key=lambda p: p["bet"])["seat"]
            ahead = sum(0 < (p["seat"] - obs["hero"]) % seat_count <= (bb_seat - obs["hero"]) % seat_count and p["stack"] > 0 for p in live)
            open_width = 0.40 if ahead >= 5 else 0.55 if ahead >= 3 else 0.76 if ahead == 2 else 0.90
            raised = obs["current_bet"] > bb
            raises = sum(e["street"] == "preflop" and e["action"] == "raise" for e in obs["history"])
            if not raised:
                if percentile <= open_width and obs["can_raise"]:
                    return raise_to(obs, obs["min_raise_to"])
                return call(obs) if cost == 0 else passive(obs)
            blocker = any(c // 4 == 12 for c in obs["hole_cards"])
            can_3bet = percentile <= 0.06 or (blocker and percentile <= 0.35 and self.rng.random() < 0.18)
            if obs["can_raise"] and raises <= 1 and can_3bet:
                return raise_to(obs, max(obs["min_raise_to"], 2.4 * obs["current_bet"]))
            odds = cost / (pot + cost) if cost else 0
            if obs["current_bet"] >= 15 * bb:
                return call(obs) if percentile <= 0.09 else passive(obs)
            if percentile <= 0.33 and odds < 0.34 and cost < 0.22 * hero["stack"]:
                return call(obs)
            return passive(obs)
        f, category, private_pair, strong, draw = _board_profile(obs)
        odds = cost / (pot + cost) if cost else 0
        heads_up = len(live) <= 2
        prior_raise = any(e["street"] == obs["street"] and e["action"] == "raise" for e in obs["history"])
        good_pair = f["overpair"] or (f["top_pair"] and f["kicker"] >= 10)
        if strong and obs["can_raise"]:
            return raise_to(obs, obs["current_bet"] + max(bb, 0.72 * (pot + cost)))
        if cost == 0:
            frequency = 0.78 if heads_up and not f["wet"] else 0.48 if heads_up else 0.22
            if obs["can_raise"] and (good_pair or draw or (not prior_raise and self.rng.random() < frequency)):
                size = 0.32 if not f["wet"] and not strong else 0.55
                return raise_to(obs, max(bb, size * pot))
            return call(obs)
        if draw and obs["can_raise"] and heads_up and not prior_raise and self.rng.random() < 0.22:
            return raise_to(obs, obs["current_bet"] + max(bb, 0.65 * (pot + cost)))
        if (good_pair and odds <= 0.34) or (private_pair and heads_up and odds <= 0.23) or (draw and odds <= 0.23):
            return call(obs)
        if heads_up and category == 0 and f["high"] == 14 and odds < 0.17 and len(obs["board"]) == 3:
            return call(obs)
        return passive(obs)


BASELINES = {
    "checkfold": CheckFold,
    "station": CallingStation,
    "tight": TightValue,
    "aggressive": Aggressive,
    "random": RandomLegal,
    "equity": UniformEquity,
    "shovewide": ShoveWide,
    "premiumnit": PremiumNit,
    "minpressure": MinraisePressure,
}


def make_baseline(name, seed=1):
    if name == "checkfold":
        return CheckFold()
    return BASELINES[name](seed=seed)
