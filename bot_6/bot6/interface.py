"""Explicit LOCAL interface. It is not the unreleased official Agent interface."""
from dataclasses import dataclass
from .cards import parse_card


@dataclass(slots=True)
class Player:
    seat: int
    stack: int
    bet: int = 0
    contributed: int = 0
    folded: bool = False
    all_in: bool = False


@dataclass(slots=True)
class State:
    hero: int
    hole: tuple
    board: tuple
    players: tuple
    button: int
    big_blind: int
    pot: int
    to_call: int
    min_raise_to: int
    max_raise_to: int
    legal: frozenset
    history: tuple = ()
    hand_id: object = 0
    table_id: object = 0
    payouts: tuple = ()
    risk_premium: float = 0.0

    @classmethod
    def from_dict(cls, obj):
        players = tuple(Player(int(p["seat"]), int(p["stack"]), int(p.get("bet", 0)),
                               int(p.get("contributed", p.get("bet", 0))),
                               bool(p.get("folded", False)), bool(p.get("all_in", int(p["stack"]) == 0)))
                        for p in obj["players"])
        hero = int(obj["hero"])
        if len({p.seat for p in players}) != len(players) or not 2 <= len(players) <= 8:
            raise ValueError("Expected 2–8 players with distinct seats")
        me = next((p for p in players if p.seat == hero), None)
        if me is None or any(min(p.stack, p.bet, p.contributed) < 0 for p in players):
            raise ValueError("Invalid hero seat or negative chips")
        hole, board = tuple(map(parse_card, obj["hole_cards"])), tuple(map(parse_card, obj.get("board", ())))
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5) or len(set(hole + board)) != len(hole + board):
            raise ValueError("Expected two hole cards, 0/3/4/5 board cards, all distinct")
        bb, pot = int(obj["big_blind"]), int(obj["pot"])
        if bb <= 0 or pot < 0:
            raise ValueError("big_blind must be positive and pot nonnegative")
        highest = max(p.bet for p in players)
        call = min(me.stack, max(0, int(obj.get("to_call", highest - me.bet))))
        maximum = min(me.bet + me.stack, int(obj.get("max_raise_to", me.bet + me.stack)))
        minimum = int(obj.get("min_raise_to", max(bb, highest + bb)))
        legal = set(obj.get("legal_actions", ("fold", "call", "raise") if call else ("check", "raise")))
        if call:
            legal.discard("check")
        else:
            legal.discard("call")
        if not obj.get("can_raise", True) or maximum <= highest or me.stack <= call:
            legal.discard("raise")
        if not legal <= {"fold", "check", "call", "raise"} or not legal:
            raise ValueError("Invalid legal_actions")
        payouts = tuple(float(x) for x in obj.get("payouts", ()))
        if payouts and (len(payouts) != len(players) or min(payouts) < 0 or any(a < b for a, b in zip(payouts, payouts[1:]))):
            raise ValueError("payouts must descend and match the remaining player count")
        return cls(hero, hole, board, players, int(obj["button"]), bb, pot, call, minimum, maximum,
                   frozenset(legal), tuple(obj.get("history", ())), obj.get("hand_id", 0), obj.get("table_id", 0),
                   payouts, max(0.0, min(.25, float(obj.get("risk_premium", 0)))))

    @property
    def me(self):
        return next(p for p in self.players if p.seat == self.hero)

    @property
    def opponents(self):
        return tuple(p for p in self.players if p.seat != self.hero and not p.folded)

    @property
    def street(self):
        return ("preflop", "", "", "flop", "turn", "river")[len(self.board)]


def legalize(state, action, target=0):
    """Raise amount is the TOTAL hero contribution on the current street."""
    if action == "raise" and "raise" in state.legal:
        # A short all-in below a full minimum raise is legal only when the
        # engine includes raise in legal_actions. Reopening is engine-owned.
        target = min(state.max_raise_to, max(state.min_raise_to, round(target)))
        if target > max(p.bet for p in state.players):
            return {"action": "raise", "amount": target}
    if action in state.legal and action != "raise":
        return {"action": action}
    for fallback in ("check", "fold", "call"):
        if fallback in state.legal:
            return {"action": fallback}
    return {"action": "raise", "amount": state.max_raise_to}


def eligible_pots(state):
    """Pot after calling, capped at hero's total contribution, and layers.

    Layers are (chips, opponent eligibility bitmask), ordered like
    state.opponents. Unknown/missing total contributions fall back to one
    pot; supplying accurate contributed values is required for side pots.
    Players still able to act are conservatively included in future layers.
    """
    opponents = state.opponents
    all_mask = (1 << len(opponents)) - 1
    if sum(p.contributed for p in state.players) != state.pot:
        return state.pot + state.to_call, None
    cap = state.me.contributed + state.to_call
    if cap <= 0:
        return state.pot + state.to_call, None
    amounts = [(p, min(cap, p.contributed + (state.to_call if p.seat == state.hero else 0))) for p in state.players]
    levels = sorted({value for _, value in amounts if value > 0})
    previous, total = 0, 0
    layers = []
    for level in levels:
        chips = (level - previous) * sum(value >= level for _, value in amounts)
        previous = level
        mask = 0
        for i, p in enumerate(opponents):
            if p.contributed >= level or not p.all_in:
                mask |= 1 << i
        layers.append((chips, mask))
        total += chips
    if all(mask == all_mask for _, mask in layers):
        return total, None
    return total, layers

