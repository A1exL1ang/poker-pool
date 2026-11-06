"""Strict validation for the *local* v1 contract, outside the timed decision path."""
from .cards import card_id


def validate_observation(o):
    required = ("hero_seat", "button", "hole_cards", "board", "players", "pot", "big_blind", "street", "hand_id", "min_raise_to", "max_raise_to", "legal_actions")
    missing = [k for k in required if k not in o]
    if missing:
        raise ValueError("missing observation fields: " + ", ".join(missing))
    board = o["board"]
    if o["street"] not in ("preflop", "flop", "turn", "river"):
        raise ValueError("unknown street")
    if len(board) != {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[o["street"]]:
        raise ValueError("board length does not match street")
    if len(o["hole_cards"]) != 2:
        raise ValueError("exactly two private cards required")
    cards = [card_id(c) for c in (*o["hole_cards"], *board)]
    if len(set(cards)) != len(cards):
        raise ValueError("duplicate visible cards")
    players = o["players"]
    if not 2 <= len(players) <= 8 or len({p["seat"] for p in players}) != len(players):
        raise ValueError("expected 2..8 uniquely identified seats")
    if o["hero_seat"] not in {p["seat"] for p in players}:
        raise ValueError("hero seat absent")
    if o["pot"] < 0 or o["big_blind"] <= 0:
        raise ValueError("negative pot or nonpositive big blind")
    for p in players:
        if p["stack"] < 0 or p["bet"] < 0 or p.get("committed", p["bet"]) < p["bet"]:
            raise ValueError("invalid player chips")
    legal = set(o["legal_actions"])
    if not legal or not legal <= {"fold", "check", "call", "raise"}:
        raise ValueError("invalid legal action names")
    hero = next(p for p in players if p["seat"] == o["hero_seat"])
    current = o.get("current_bet", max(p["bet"] for p in players))
    call = min(hero["stack"], max(0, current - hero["bet"]))
    if "to_call" in o and o["to_call"] != call:
        raise ValueError("to_call must be the capped additional call amount")
    if "check" in legal and call > 0:
        raise ValueError("checking offered while facing a bet")
    if "raise" in legal and o["max_raise_to"] > hero["bet"] + hero["stack"]:
        raise ValueError("raise exceeds hero chips")
    return o


def validate_action(o, action):
    name, amount = action.get("action"), action.get("amount")
    if name not in o["legal_actions"] or not isinstance(amount, int) or isinstance(amount, bool):
        raise ValueError("action is not offered or amount is not an integer")
    h = next(p for p in o["players"] if p["seat"] == o["hero_seat"])
    if name in ("fold", "check") and amount != 0:
        raise ValueError("fold/check amount must be zero")
    if name == "call":
        current = o.get("current_bet", max(p["bet"] for p in o["players"]))
        call = min(h["stack"], max(0, current - h["bet"]))
        if amount != call:
            raise ValueError("call amount must be the additional chips paid")
    if name == "raise":
        if not h["bet"] < amount <= o["max_raise_to"]:
            raise ValueError("raise-to outside permitted range")
        if amount < o["min_raise_to"] and amount != h["bet"] + h["stack"]:
            raise ValueError("only an all-in may be below the minimum full raise")
        if amount <= o.get("current_bet", max(p["bet"] for p in o["players"])):
            raise ValueError("raise must increase the current highest bet")
    return action
