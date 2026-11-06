"""Explicit LOCAL observation/action contract; not an official contest API.

All amounts are nonnegative integer chips. ``stacks`` are chips behind, whereas
``committed`` includes ALL chips paid into this hand. ``pot`` includes current
street wagers, including the wager being faced. ``to_call`` is the UNcapped
nominal deficit. A call pays min(stack, to_call). Raise amounts are final street
totals (raise-TO), not increments. ``min_raise_to`` can exceed ``max_raise_to``
only when the sole permitted raise is a short all-in. The engine supplies
``can_raise``; the policy never tries to infer reopening rights.

Cards are integer IDs 0..51: rank = card // 4 + 2, suit = card % 4.
Street 0/1/2/3 = preflop/flop/turn/river. Seats are stable anonymous indices.
Hole cards of opponents and future board cards must NEVER be passed to Agent.
"""

from typing import Literal, NotRequired, TypedDict


class Action(TypedDict):
    action: Literal["fold", "check", "call", "raise"]
    amount: int


class HistoryEvent(TypedDict):
    seat: int
    street: int
    action: str
    amount: int
    to_call: int
    pot_before: int


class Observation(TypedDict):
    hero: int
    hand_id: int
    street: int
    button: int
    sb_seat: int
    bb_seat: int
    small_blind: int
    big_blind: int
    stacks: list[int]
    hand_start_stacks: list[int]
    committed: list[int]
    street_committed: list[int]
    folded: list[bool]
    all_in: list[bool]
    hole_cards: list[int]
    board: list[int]
    pot: int
    to_call: int
    min_raise_to: int
    max_raise_to: int
    can_raise: bool
    active_players: int
    history: list[HistoryEvent]
    payouts: NotRequired[list[float] | None]
    hands_remaining: NotRequired[int]
    table_id: NotRequired[int | str]


def passive(obs: Observation) -> Action:
    """The same conservative fallback as the stated tournament timeout rule."""
    return {"action": "check" if obs.get("to_call", 0) <= 0 else "fold", "amount": 0}


def call(obs: Observation) -> Action:
    if obs["to_call"] <= 0:
        return {"action": "check", "amount": 0}
    return {"action": "call", "amount": min(obs["to_call"], obs["stacks"][obs["hero"]])}


def raise_to(obs: Observation, target: float) -> Action:
    """Clamp a desired raise to explicitly provided legal bounds."""
    if not obs["can_raise"]:
        return call(obs)
    hi = int(obs["max_raise_to"])
    lo = int(obs["min_raise_to"])
    amount = min(hi, max(lo, int(target + 0.5)))
    current = obs["street_committed"][obs["hero"]]
    if amount <= current + obs["to_call"]:
        return call(obs)
    return {"action": "raise", "amount": amount}


def is_legal(obs: Observation, action: dict) -> bool:
    """Strict local contract validator, used by tests/benchmarks, not hot path."""
    if not isinstance(action, dict):
        return False
    kind = action.get("action")
    amount = action.get("amount", 0)
    if not isinstance(amount, int) or isinstance(amount, bool):
        return False
    if kind == "fold":
        return amount == 0
    if kind == "check":
        return obs["to_call"] == 0 and amount == 0
    if kind == "call":
        return obs["to_call"] > 0 and amount == min(obs["to_call"], obs["stacks"][obs["hero"]])
    if kind == "raise":
        current = obs["street_committed"][obs["hero"]]
        return (obs["can_raise"] and amount > current + obs["to_call"]
                and amount <= obs["max_raise_to"]
                and (amount >= obs["min_raise_to"] or amount == obs["max_raise_to"]))
    return False


def validate_observation(obs: Observation) -> None:
    """Expensive integration check. Raises ValueError on ambiguous/invalid data."""
    n = len(obs["stacks"])
    if not 2 <= n <= 8 or not 0 <= obs["hero"] < n:
        raise ValueError("Require 2..8 fixed seats and a valid hero index")
    for key in ("hand_start_stacks", "committed", "street_committed", "folded", "all_in"):
        if len(obs[key]) != n:
            raise ValueError(f"{key} length differs from stacks")
    for key in ("stacks", "hand_start_stacks", "committed", "street_committed"):
        if any(type(x) is not int or x < 0 for x in obs[key]):
            raise ValueError(f"{key} must contain nonnegative integer chips")
    cards = list(obs["hole_cards"]) + list(obs["board"])
    if len(obs["hole_cards"]) != 2 or len(obs["board"]) != (0, 3, 4, 5)[obs["street"]]:
        raise ValueError("Board/hole length does not match street")
    if len(set(cards)) != len(cards) or any(type(c) is not int or not 0 <= c < 52 for c in cards):
        raise ValueError("Invalid or duplicate visible card")
    if obs["pot"] != sum(obs["committed"]):
        raise ValueError("pot must include all current and previous wagers")
    if any(s > c for s, c in zip(obs["street_committed"], obs["committed"])):
        raise ValueError("Street contribution exceeds total contribution")
    h = obs["hero"]
    if obs["folded"][h] or obs["stacks"][h] <= 0:
        raise ValueError("Hero cannot act when folded or all-in")
    if obs["to_call"] < 0 or obs["big_blind"] <= 0:
        raise ValueError("Invalid call or blind amount")
    payouts = obs.get("payouts")
    if payouts is not None and (len(payouts) != n or any(x < 0 for x in payouts)
                               or any(a < b for a, b in zip(payouts, payouts[1:]))):
        raise ValueError("Payouts must be nonnegative, nonincreasing, one per starting seat")
