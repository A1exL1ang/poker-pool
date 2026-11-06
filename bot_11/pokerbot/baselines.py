"""Transparent, deliberately simple local opponents, never official opponents.

These provide different exploitation and variance tests. They are not evidence
of performance against unknown contest entrants, and are not pretrained bots.
"""
from __future__ import annotations

from collections import Counter
import random


def preflop_strength(cards):
    """A crude ordering in [0,1], NOT an equity estimate."""
    a, b = sorted((c // 4 for c in cards), reverse=True)
    if a == b:
        return .55 + .035 * a
    suited = cards[0] % 4 == cards[1] % 4
    gap = a - b
    return min(.94, .16 + .035 * a + .018 * b + .055 * suited + .035 * (gap == 1) - .02 * max(0, gap - 3))


def postflop_strength(cards, board):
    """Crude made-hand ordering used only for local baseline diversity."""
    hole = [c // 4 for c in cards]
    ranks = Counter(c // 4 for c in cards + board)
    branks = Counter(c // 4 for c in board)
    suits = Counter(c % 4 for c in cards + board)
    unique = set(ranks)
    if 12 in unique:
        unique.add(-1)
    straight = any(all(r in unique for r in range(low, low + 5)) for low in range(-1, 9))
    flush = max(suits.values()) >= 5
    counts = sorted(ranks.values(), reverse=True)
    if counts[0] == 4:
        return .99
    if counts[0] >= 3 and counts[1] >= 2:
        return .96
    if flush:
        return .91
    if straight:
        return .87
    if any(ranks[r] >= 3 and branks[r] < ranks[r] for r in hole):
        return .83
    own_pairs = sum(ranks[r] >= 2 and branks[r] < ranks[r] for r in set(hole))
    if own_pairs >= 2:
        return .78
    high_board = max(branks)
    if hole[0] == hole[1]:
        return .75 if hole[0] > high_board else .47 + .015 * hole[0]
    matched = [r for r in hole if branks[r]]
    if matched:
        paired = max(matched)
        return .72 if paired == high_board else .48 + .012 * paired
    draw = max(suits.values()) == 4 and len(board) < 5
    return .19 + .017 * max(hole) + .17 * draw


def _raise(obs, target):
    if not obs["can_raise"]:
        return {"action": "call" if obs["to_call"] else "check"}
    maximum = obs["max_raise_to"]
    minimum = min(maximum, obs["min_raise_to"])
    return {"action": "raise", "amount": min(maximum, max(minimum, int(target)))}


class FoldAgent:
    def __call__(self, obs):
        return {"action": "fold" if obs["to_call"] else "check"}

    act = __call__


class CallingStation:
    def __call__(self, obs):
        return {"action": "call" if obs["to_call"] else "check"}

    act = __call__


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __call__(self, obs):
        value = self.rng.random()
        if value < .15 and obs["to_call"]:
            return {"action": "fold"}
        if value > .70 and obs["can_raise"]:
            hero = obs["players"][obs["seat"]]
            return _raise(obs, hero["street_bet"] + obs["to_call"] + max(obs["big_blind"], obs["pot"] * (.35 + value)))
        return {"action": "call" if obs["to_call"] else "check"}

    act = __call__


class StyleAgent:
    """Fixed thresholds without table adaptation or payout modelling."""
    def __init__(self, seed=0, looseness=0.0, aggression=0.0):
        self.rng = random.Random(seed)
        self.looseness = looseness
        self.aggression = aggression

    def __call__(self, obs):
        hero = obs["players"][obs["seat"]]
        stack = hero["stack"]
        call = obs["to_call"]
        pot = obs["pot"]
        bb = obs["big_blind"]
        preflop = not obs["board"]
        value = preflop_strength(obs["hole_cards"]) if preflop else postflop_strength(obs["hole_cards"], obs["board"])
        value += self.looseness
        price = call / max(1, pot + call)
        if preflop:
            raised = obs["current_bet"] > bb
            if value > .89 or (stack < 8 * bb and value > .63):
                return _raise(obs, obs["max_raise_to"])
            if obs["can_raise"] and value > (.73 if raised else .60) - self.aggression:
                return _raise(obs, max(3 * bb, obs["current_bet"] * 3))
            threshold = .47 + min(.29, call / max(bb, stack + call)) + (.08 if raised else 0)
            if call and value < threshold:
                return {"action": "fold"}
        else:
            if obs["can_raise"] and (value > .77 - self.aggression or (value < .35 and self.rng.random() < max(0, self.aggression))):
                return _raise(obs, hero["street_bet"] + call + max(bb, .70 * (pot + call)))
            if call and value < price + .28:
                return {"action": "fold"}
        return {"action": "call" if call else "check"}

    act = __call__


class TightPassive(StyleAgent):
    def __init__(self, seed=0):
        super().__init__(seed, looseness=-.08, aggression=-.12)


class TightAggressive(StyleAgent):
    def __init__(self, seed=0):
        super().__init__(seed, looseness=-.015, aggression=.04)


class LooseAggressive(StyleAgent):
    def __init__(self, seed=0):
        super().__init__(seed, looseness=.11, aggression=.14)


class PushFold:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __call__(self, obs):
        if obs["board"]:
            return {"action": "call" if obs["to_call"] else "check"}
        strength = preflop_strength(obs["hole_cards"])
        hero = obs["players"][obs["seat"]]
        depth = (hero["stack"] + hero["street_bet"]) / obs["big_blind"]
        threshold = min(.88, .43 + depth * .009)
        if strength >= threshold:
            return _raise(obs, obs["max_raise_to"])
        return {"action": "fold" if obs["to_call"] else "check"}

    act = __call__


BASELINES = {"fold": FoldAgent, "station": CallingStation, "random": RandomAgent,
             "tight_passive": TightPassive, "tag": TightAggressive,
             "lag": LooseAggressive, "pushfold": PushFold}


def make_baseline(name, seed=0):
    cls = BASELINES[name]
    return cls() if name in ("fold", "station") else cls(seed)
