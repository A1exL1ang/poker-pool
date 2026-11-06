"""Explicit LOCAL interface; the unavailable contest interface is not emulated.

See README.md for field definitions. The adapter is deliberately narrow so an
official integration can be reviewed without guessing chip/action semantics.
"""
RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card(value):
    """Translate a local integer or a two-character card such as 'As'."""
    if isinstance(value, int):
        if 0 <= value < 52:
            return value
        raise ValueError("card must be between 0 and 51")
    if len(value) == 2:
        return 4 * RANKS.index(value[0].upper()) + SUITS.index(value[1].lower())
    raise ValueError("card must be a local integer or rank/suit string")


def card_name(value):
    return RANKS[value >> 2] + SUITS[value & 3]


def passive(observation):
    return {"action": "check" if observation.get("to_call", 0) <= 0 else "fold"}


def legalize(observation, action):
    """Normalize *our own* intent to legal local actions, without silent calling.

    The engine independently validates this output. A raise amount is a total
    street commitment, never an increment. Short all-in raises are permitted.
    """
    kind = action.get("action")
    call = max(0, observation["to_call"])
    if kind in ("fold", "check"):
        return {"action": "check" if not call else "fold"}
    if kind == "call":
        return {"action": "call" if call else "check"}
    if kind != "raise" or not observation.get("can_raise", False):
        return passive(observation)
    p = observation["players"][observation["hero"]]
    maximum = p["stack"] + p["committed"]
    if maximum <= observation["current_bet"]:
        return {"action": "call" if call else "check"}
    amount = min(maximum, max(int(action["amount"]), observation["min_raise_to"]))
    return {"action": "raise", "amount": amount}
