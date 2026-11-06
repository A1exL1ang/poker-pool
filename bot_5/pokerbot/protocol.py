"""Our local interface, NOT an official competition adapter.

See README.md for units, pot semantics, and an example observation.
"""
from .cards import parse_card


def safe_action(observation):
    legal = observation.get("legal_actions", {}) if isinstance(observation, dict) else {}
    if not isinstance(legal, dict):
        legal = {}
    if legal.get("check"):
        return {"action": "check", "amount": 0}
    if legal.get("fold", True):
        return {"action": "fold", "amount": 0}
    if "call" in legal:
        return {"action": "call", "amount": max(0, int(legal["call"]))}
    return {"action": "check", "amount": 0}


def normalize(obs):
    hole = [parse_card(c) for c in obs["hole_cards"]]
    board = [parse_card(c) for c in obs.get("board", [])]
    if len(hole) != 2 or len(board) not in (0, 3, 4, 5) or len(set(hole + board)) != len(hole + board):
        raise ValueError("expected two private cards and 0/3/4/5 unique board cards")
    players = obs["players"]
    hero = int(obs["hero"])
    if not 2 <= len(players) <= 8 or not 0 <= hero < len(players):
        raise ValueError("invalid seats")
    if obs["big_blind"] <= 0 or obs["pot"] < 0:
        raise ValueError("invalid blind or pot")
    for p in players:
        if p["stack"] < 0 or p.get("bet", 0) < 0 or p.get("committed", p.get("bet", 0)) < p.get("bet", 0):
            raise ValueError("invalid chip accounting")
    if players[hero].get("folded") or players[hero]["stack"] <= 0:
        raise ValueError("hero cannot act")
    legal = obs["legal_actions"]
    if not isinstance(legal, dict):
        raise ValueError("legal_actions must be an explicit mapping")
    if "call" in legal and not 0 < legal["call"] <= players[hero]["stack"]:
        raise ValueError("call must be an incremental positive amount, capped by stack")
    if "raise" in legal:
        r = legal["raise"]
        maximum = players[hero]["stack"] + players[hero].get("bet", 0)
        if not 0 < r["min"] <= r["max"] <= maximum:
            raise ValueError("invalid raise-to bounds")
        if r["min"] <= players[hero].get("bet", 0) + legal.get("call", 0):
            raise ValueError("raise-to must exceed a call")
    return hole, board, players, hero, legal


def legalize(action, amount, legal):
    if action == "raise" and "raise" in legal:
        r = legal["raise"]
        return {"action": "raise", "amount": max(int(r["min"]), min(int(r["max"]), int(round(amount))))}
    if action == "call" and "call" in legal:
        return {"action": "call", "amount": int(legal["call"])}
    if legal.get("check"):
        return {"action": "check", "amount": 0}
    if action == "fold" and legal.get("fold", True):
        return {"action": "fold", "amount": 0}
    if "call" in legal and action in ("raise", "call"):
        return {"action": "call", "amount": int(legal["call"])}
    return {"action": "fold", "amount": 0}

