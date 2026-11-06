"""Transparent, independently authored stress opponents, not trained rivals.

These intentionally varied heuristics are local comparators only. Beating them
does not establish equilibrium play or performance against the actual field.
"""

import random

from pokerbot.evaluator import evaluate
from pokerbot.interface import call, passive, raise_to


def preflop_strength(hole):
    """Coarse heads-up strength proxy, deliberately independent of bot assets."""
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    if a == b:
        return .49 + .029 * (a - 2)
    suited = hole[0] % 4 == hole[1] % 4
    gap = a - b
    return min(.78, .30 + .018 * (a - 2) + .012 * (b - 2)
               + .025 * (a == 14) + .025 * suited - .005 * gap
               + .015 * (gap == 1 and a < 12))


def made_strength(hole, board):
    if not board:
        return preflop_strength(hole)
    value = evaluate(hole + board)
    category = value >> 20
    high = (value >> 16) & 15
    hole_ranks = [c // 4 + 2 for c in hole]
    board_ranks = [c // 4 + 2 for c in board]
    if category == 0:
        strength = .15 + .020 * (max(hole_ranks) - 2)
    elif category == 1:
        own_pair = high in hole_ranks
        strength = (.52 + .015 * high if own_pair else .27 + .01 * max(hole_ranks))
        if own_pair and high >= max(board_ranks):
            strength += .09
    elif category == 2:
        strength = .83 if any(r in board_ranks for r in hole_ranks) else .60
    elif category == 3:
        strength = .93 if high in hole_ranks else .48 + .015 * max(hole_ranks)
    else:
        strength = (.95, .96, .985, .995, 1.0)[category - 4]
    # Board-made monsters must not be mistaken for private nuts.
    if len(board) == 5 and value == evaluate(board):
        strength = min(strength, .55)
    if len(board) < 5:
        suit_counts = [sum(c % 4 == suit for c in hole + board) for suit in range(4)]
        if max(suit_counts) == 4 and any(sum(c % 4 == s for c in hole) for s in range(4) if suit_counts[s] == 4):
            strength = min(.90, strength + (.13 if len(board) == 3 else .07))
        ranks = set(hole_ranks + board_ranks)
        if 14 in ranks:
            ranks.add(1)
        if max(len(ranks.intersection(range(low, low + 5))) for low in range(1, 11)) == 4:
            strength = min(.90, strength + (.08 if len(board) == 3 else .04))
    return max(.02, min(.999, strength))


class Baseline:
    name = "baseline"

    def __init__(self, seed=0):
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)

    def __call__(self, obs):
        return self.act(obs)


class CheckFold(Baseline):
    name = "checkfold"

    def act(self, obs):
        return passive(obs)


class CallingStation(Baseline):
    name = "station"

    def act(self, obs):
        strength = made_strength(obs["hole_cards"], obs["board"])
        if obs["can_raise"] and strength > (.80 if obs["street"] == 0 else .95):
            return raise_to(obs, obs["street_committed"][obs["hero"]] + obs["to_call"] + max(obs["big_blind"], obs["pot"] * .7))
        if obs["to_call"] == 0:
            return call(obs)
        cost = min(obs["to_call"], obs["stacks"][obs["hero"]])
        threshold = cost / (obs["pot"] + cost)
        multi = max(1, obs["active_players"] - 1)
        equity = strength ** (.42 + .28 * multi)
        return call(obs) if equity > threshold + .015 else passive(obs)


class Nit(Baseline):
    name = "nit"

    def act(self, obs):
        strength = made_strength(obs["hole_cards"], obs["board"])
        hero = obs["hero"]
        if obs["street"] == 0:
            if strength > .71:
                return raise_to(obs, max(3 * obs["big_blind"], 3 * (obs["street_committed"][hero] + obs["to_call"])))
            if obs["to_call"] <= obs["big_blind"] and strength > .59:
                return call(obs)
            return passive(obs)
        if strength > .90 and obs["can_raise"]:
            return raise_to(obs, obs["street_committed"][hero] + obs["to_call"] + obs["pot"] * .65)
        cost = min(obs["to_call"], obs["stacks"][hero])
        threshold = cost / max(1, obs["pot"] + cost)
        return call(obs) if not cost or strength ** max(1, obs["active_players"] - 1) > threshold + .26 else passive(obs)


class TAG(Baseline):
    name = "tag"

    def act(self, obs):
        hero = obs["hero"]
        strength = made_strength(obs["hole_cards"], obs["board"])
        cost = min(obs["to_call"], obs["stacks"][hero])
        current = obs["street_committed"][hero] + obs["to_call"]
        if obs["street"] == 0:
            remaining = sum(not obs["folded"][i] and i != hero for i in range(len(obs["stacks"])))
            opening = not any(e["action"] == "raise" for e in obs["history"])
            threshold = .48 + .018 * remaining
            if opening and strength > threshold:
                return raise_to(obs, max(2.5 * obs["big_blind"], current + 2 * obs["big_blind"]))
            if strength > .745:
                return raise_to(obs, max(current * 3, current + obs["big_blind"]))
            if (cost <= 3 * obs["big_blind"] and strength > .57) or (strength > .66 and cost <= obs["stacks"][hero] * .15):
                return call(obs)
            return passive(obs)
        multi = max(1, obs["active_players"] - 1)
        equity = strength ** (.75 + .45 * multi)
        threshold = cost / max(1, obs["pot"] + cost)
        if obs["can_raise"]:
            bluff = not cost and multi == 1 and self.rng.random() < .14
            if equity > (.78 if cost else .64) or bluff:
                return raise_to(obs, current + obs["pot"] * (.75 if equity > .85 else .5))
        return call(obs) if not cost or equity > threshold + .11 else passive(obs)


class Maniac(Baseline):
    name = "maniac"

    def act(self, obs):
        strength = made_strength(obs["hole_cards"], obs["board"])
        roll = self.rng.random()
        hero = obs["hero"]
        cost = min(obs["to_call"], obs["stacks"][hero])
        if obs["can_raise"] and (roll < .48 or strength > .80):
            if roll < .10 or obs["stacks"][hero] < obs["pot"] * 2:
                target = obs["max_raise_to"]
            else:
                target = obs["street_committed"][hero] + cost + max(obs["big_blind"] * 2, obs["pot"] * 1.1)
            return raise_to(obs, target)
        return call(obs) if not cost or strength > .45 or roll < .32 else passive(obs)


class RandomLegal(Baseline):
    name = "random"

    def act(self, obs):
        roll = self.rng.random()
        if obs["can_raise"] and roll < .25:
            if self.rng.random() < .08:
                target = obs["max_raise_to"]
            else:
                target = obs["min_raise_to"] + self.rng.random() * min(obs["pot"] * 2, obs["max_raise_to"])
            return raise_to(obs, target)
        if not obs["to_call"] or roll < .78:
            return call(obs)
        return passive(obs)


class EquityHeuristic(Baseline):
    """Pot-odds policy using its own coarse hand-class strength approximation."""
    name = "equity"

    def act(self, obs):
        hero = obs["hero"]
        strength = made_strength(obs["hole_cards"], obs["board"])
        opponents = max(1, obs["active_players"] - 1)
        if obs["street"] == 0:
            strength = min(.99, strength + .045)
        equity = strength ** (1 + .48 * (opponents - 1))
        cost = min(obs["to_call"], obs["stacks"][hero])
        threshold = cost / max(1, obs["pot"] + cost)
        if obs["can_raise"] and (equity > .68 or (not cost and equity > .50)):
            current = obs["street_committed"][hero] + obs["to_call"]
            return raise_to(obs, current + max(obs["big_blind"] * 2, obs["pot"] * .7))
        return call(obs) if not cost or equity > threshold + .07 else passive(obs)


class ModelEquity(Baseline):
    """Matched-model control: pot odds/value bets, no ranges, adaptation, or ICM.

    Uses the candidate's learned equity estimator to separate its estimation
    benefit from its strategic/accounting additions. This is deliberately a
    different comparator from the independently authored coarse ``equity``.
    """
    name = "model_equity"

    def __init__(self, seed=0):
        from pokerbot.equity import EquityEstimator
        self.estimator = EquityEstimator()
        super().__init__(seed)

    def act(self, obs):
        hero = obs["hero"]
        opponents = max(1, obs["active_players"] - 1)
        eq = self.estimator.estimate(obs["hole_cards"], obs["board"], opponents, 1.0, 0.0)
        cost = min(obs["to_call"], obs["stacks"][hero])
        cap = obs["committed"][hero] + cost
        eligible = sum(min(x, cap) for x in obs["committed"]) + cost
        price = cost / max(1, eligible)
        if obs["can_raise"] and eq > max(.44, 1 / (opponents + 1) + .17):
            current = obs["street_committed"][hero] + obs["to_call"]
            return raise_to(obs, current + max(2 * obs["big_blind"], obs["pot"] * .65))
        return call(obs) if not cost or eq > price + .035 else passive(obs)


BASELINES = {cls.name: cls for cls in (CheckFold, CallingStation, Nit, TAG, Maniac, RandomLegal, EquityHeuristic, ModelEquity)}


def make_baseline(name, seed=0):
    return BASELINES[name](seed=seed)
