"""The explicit LOCAL contract, independent of any unavailable official SDK."""

from .cards import card

STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def safe_action(observation):
    """Match the contest's stated error fallback for our canonical observations."""
    try:
        return {"action": "check" if int(observation.get("to_call", 0)) <= 0 else "fold"}
    except (ValueError, TypeError, AttributeError, OverflowError):
        return {"action": "fold"}


def legalize(action, observation):
    """Normalize a strategy decision to legal canonical call/check/fold/raise-to."""
    fallback = safe_action(observation)
    if not isinstance(action, dict):
        return fallback
    kind = action.get("action")
    call = max(0, int(observation["to_call"]))
    if kind in ("call", "check"):
        return {"action": "call" if call else "check"}
    if kind == "fold":
        return fallback
    if kind != "raise" or not observation.get("can_raise", False):
        return fallback
    low = int(observation["min_raise_to"])
    high = int(observation["max_raise_to"])
    current = int(observation["current_bet"])
    if high <= current:
        return fallback
    try:
        amount = int(action["amount"])
    except (ValueError, TypeError, KeyError, OverflowError):
        return fallback
    # The sole legal below-minimum raise is a short all-in to the maximum.
    amount = high if high < low else min(high, max(low, amount))
    return {"action": "raise", "amount": amount}


def validate(observation):
    """Strict opt-in validation for adapters/tests; excluded from the hot path."""
    required = ("hand_id", "street", "hero", "button", "big_blind", "hole_cards",
                "board", "pot", "to_call", "current_bet", "min_raise_to",
                "max_raise_to", "can_raise", "players")
    for key in required:
        if key not in observation:
            raise ValueError("missing observation field: " + key)
    if observation["street"] not in STREETS:
        raise ValueError("invalid street")
    hole = tuple(card(c) for c in observation["hole_cards"])
    board = tuple(card(c) for c in observation["board"])
    if len(hole) != 2 or len(board) != (0, 3, 4, 5)[STREETS[observation["street"]]]:
        raise ValueError("hole/board length is inconsistent with street")
    if len(set(hole + board)) != len(hole + board):
        raise ValueError("duplicate visible cards")
    players = observation["players"]
    if not 2 <= len(players) <= 8:
        raise ValueError("local interface supports 2..8 occupied seats")
    if len({p["seat"] for p in players}) != len(players):
        raise ValueError("duplicate seat")
    hero = next((p for p in players if p["seat"] == observation["hero"]), None)
    if hero is None or hero.get("folded") or hero.get("all_in"):
        raise ValueError("hero cannot act")
    for p in players:
        if any(int(p.get(k, 0)) < 0 for k in ("stack", "bet", "committed")):
            raise ValueError("negative chip count")
        if p["bet"] > p["committed"]:
            raise ValueError("street bet exceeds hand commitment")
    if int(observation["pot"]) != sum(p["committed"] for p in players):
        raise ValueError("pot includes all hand commitments and current bets")
    if observation["to_call"] != min(hero["stack"], max(0, observation["current_bet"] - hero["bet"])):
        raise ValueError("to_call must be capped to hero's stack")
    if observation["max_raise_to"] != hero["stack"] + hero["bet"]:
        raise ValueError("max_raise_to is hero's total street capacity")
    return True
