"""Locally authored, deliberately simple benchmark opponents.

These are behavioral stress tests, not established strong poker programs and
not evidence of an equilibrium strategy. No opponent identity information is
used by the candidate bot.
"""
import random

from simulator import evaluate


def preflop_strength(hole):
    a, b = sorted((c // 4 for c in hole), reverse=True)
    suited = hole[0] % 4 == hole[1] % 4
    if a == b:
        return 0.49 + 0.029 * a
    gap = a - b - 1
    value = 0.22 + 0.025 * a + 0.014 * b
    value += 0.035 * suited + 0.02 * (gap == 0) - 0.013 * max(0, gap - 1)
    if a == 12:
        value += 0.035
    return max(0.16, min(0.84, value))


def hand_strength(obs):
    hole, board = obs["hole"], obs["board"]
    if not board:
        return preflop_strength(hole)
    rank = evaluate(hole + board)
    category = rank[0]
    if category >= 4:
        strength = (0.91, 0.95, 0.97, 0.995, 0.999)[category - 4]
    elif category == 3:
        strength = 0.9
    elif category == 2:
        strength = 0.8
    elif category == 1:
        board_ranks = [c // 4 for c in board]
        paired = rank[1]
        if paired not in [c // 4 for c in hole]:
            strength = 0.26 + max(c // 4 for c in hole) * 0.013
        else:
            strength = 0.48 + 0.19 * (paired >= max(board_ranks)) + paired * 0.005
    else:
        strength = 0.12 + max(c // 4 for c in hole) * 0.017
    if len(board) == 5 and evaluate(board) == rank:
        strength = min(strength, 0.35)
    if len(board) < 5:
        cards = hole + board
        suits = [sum(c % 4 == s for c in cards) for s in range(4)]
        if max(suits) == 4 and any(suits[c % 4] == 4 for c in hole):
            strength += 0.12 if len(board) == 3 else 0.07
        ranks = set(c // 4 for c in cards)
        if 12 in ranks:
            ranks.add(-1)
        if max(sum(r in ranks for r in range(low, low + 5)) for low in range(-1, 9)) == 4:
            strength += 0.06 if len(board) == 3 else 0.03
    live = sum(not p["folded"] for p in obs["players"])
    return min(0.999, max(0.01, strength - 0.04 * max(0, live - 2)))


def raise_action(obs, desired):
    if "raise" not in obs["legal_actions"]:
        return {"action": "check" if "check" in obs["legal_actions"] else "call"}
    maximum = obs["max_raise_to"]
    minimum = obs["min_raise_to"]
    return {"action": "raise", "amount": min(maximum, max(minimum, int(desired)))}


class BaseAgent:
    def __init__(self, seed=0):
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)


class FoldAgent(BaseAgent):
    def act(self, obs):
        return {"action": "check" if "check" in obs["legal_actions"] else "fold"}


class CallingAgent(BaseAgent):
    def act(self, obs):
        return {"action": "check" if "check" in obs["legal_actions"] else "call"}


class RandomAgent(BaseAgent):
    def act(self, obs):
        legal = obs["legal_actions"]
        roll = self.rng.random()
        if "raise" in legal and roll < 0.20:
            p = obs["players"][obs["hero"]]
            target = p["bet"] + obs["to_call"] + max(obs["big_blind"], int(obs["pot"] * self.rng.choice((0.33, 0.7, 1.2))))
            return raise_action(obs, target)
        if "check" in legal:
            return {"action": "check"}
        return {"action": "fold" if roll < 0.40 else "call"}


class TightAgent(BaseAgent):
    def act(self, obs):
        strength = hand_strength(obs)
        call = obs["to_call"]
        pot = obs["pot"]
        hero = obs["players"][obs["hero"]]
        if obs["street"] == 0:
            facing_raise = max(p["bet"] for p in obs["players"]) > obs["big_blind"]
            threshold = 0.64 if facing_raise else 0.57
            if strength > threshold + 0.11 and "raise" in obs["legal_actions"]:
                target = max(3 * obs["big_blind"], 3 * max(p["bet"] for p in obs["players"]))
                return raise_action(obs, target)
            if "check" in obs["legal_actions"]:
                return {"action": "check"}
            if strength >= threshold and (call < hero["stack"] * 0.2 or strength > 0.76):
                return {"action": "call"}
            return {"action": "fold"}
        if strength > 0.77 and "raise" in obs["legal_actions"]:
            return raise_action(obs, hero["bet"] + call + max(obs["big_blind"], (pot + call) * 0.65))
        if not call:
            return {"action": "check"}
        odds = call / (pot + call)
        threshold = max(0.44, odds + 0.2 + 0.12 * (call > hero["stack"] * 0.4))
        return {"action": "call" if strength > threshold else "fold"}


class PressureAgent(BaseAgent):
    def act(self, obs):
        strength = hand_strength(obs)
        hero = obs["players"][obs["hero"]]
        call = obs["to_call"]
        pot = obs["pot"]
        raise_count = sum(a["action"] == "raise" and a["street"] == obs["street"] for a in obs["history"])
        if obs["street"] == 0:
            if strength > 0.65 or (strength > 0.43 and raise_count == 0):
                target = 2.5 * obs["big_blind"] if raise_count == 0 else 3 * max(p["bet"] for p in obs["players"])
                if hero["stack"] <= 12 * obs["big_blind"] and strength > 0.5:
                    target = obs["max_raise_to"]
                if "raise" in obs["legal_actions"]:
                    return raise_action(obs, target)
            if not call:
                return {"action": "check"}
            return {"action": "call" if strength > 0.51 + 0.035 * raise_count else "fold"}
        if "raise" in obs["legal_actions"]:
            probability = 0.8 if strength > 0.73 else (0.45 if not call else 0.12)
            if raise_count < 2 and self.rng.random() < probability:
                return raise_action(obs, hero["bet"] + call + max(obs["big_blind"], 0.72 * (pot + call)))
        if not call:
            return {"action": "check"}
        return {"action": "call" if strength > max(0.36, call / (pot + call) + 0.1) else "fold"}


class ShoveAgent(BaseAgent):
    def act(self, obs):
        strength = hand_strength(obs)
        if strength > (0.58 if obs["street"] == 0 else 0.66) and "raise" in obs["legal_actions"]:
            return raise_action(obs, obs["max_raise_to"])
        if "check" in obs["legal_actions"]:
            return {"action": "check"}
        return {"action": "call" if strength > 0.64 else "fold"}


BASELINES = {"fold": FoldAgent, "calling": CallingAgent, "random": RandomAgent,
             "tight": TightAgent, "pressure": PressureAgent, "shove": ShoveAgent}
