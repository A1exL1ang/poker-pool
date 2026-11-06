"""Local interface contract. This is NOT an official competition interface."""
from dataclasses import dataclass, field
from .cards import card


@dataclass(frozen=True, slots=True)
class Player:
    seat: int
    stack: int
    committed: int = 0       # This street, INCLUDING blinds but EXCLUDING antes.
    contributed: int = 0     # This whole hand, including antes.
    folded: bool = False
    all_in: bool = False


@dataclass(frozen=True, slots=True)
class Event:
    seat: int
    street: str
    action: str             # fold/check/call/raise; blinds are not events.
    amount: int = 0         # Total committed on street AFTER action.
    to_call: int = 0        # Full amount faced BEFORE action.
    pot_before: int = 0


@dataclass(frozen=True, slots=True)
class State:
    table_id: str
    hand_id: int
    hero: int
    button: int
    hole: tuple
    board: tuple
    players: tuple
    pot: int                # All chips in the middle, INCLUDING current bets.
    big_blind: int
    to_call: int             # Uncapped call cost (agent caps it at own stack).
    min_raise_to: int        # Minimum FULL raise, in total street contribution.
    max_raise_to: int        # Maximum legal raise-to, normally committed + stack.
    can_raise: bool = True   # False if action was not reopened by a short all-in.
    history: tuple = ()      # Cumulative public action history for this hand.
    payouts: tuple = ()      # Points by finish, descending, all remaining ranks.
    ante: int = 0

    @property
    def street(self):
        return ("preflop", "flop", "turn", "river")[(0, 3, 4, 5).index(len(self.board))]

    @classmethod
    def from_dict(cls, d):
        return cls(
            table_id=str(d["table_id"]), hand_id=int(d["hand_id"]),
            hero=int(d["hero"]), button=int(d["button"]),
            hole=tuple(card(c) for c in d["hole"]),
            board=tuple(card(c) for c in d.get("board", ())),
            players=tuple(Player(**p) for p in d["players"]),
            pot=int(d["pot"]), big_blind=int(d["big_blind"]),
            to_call=int(d["to_call"]), min_raise_to=int(d["min_raise_to"]),
            max_raise_to=int(d["max_raise_to"]), can_raise=bool(d.get("can_raise", True)),
            history=tuple(Event(**e) for e in d.get("history", ())),
            payouts=tuple(d.get("payouts", ())), ante=int(d.get("ante", 0)),
        )


@dataclass(frozen=True, slots=True)
class Action:
    action: str
    amount: int = 0          # Only raise has an amount: TOTAL street contribution.

    def to_dict(self):
        return {"action": self.action, "amount": self.amount}


def legalize(state, action):
    """Clamp every policy result to this explicit no-limit action contract.

    The engine remains the authority for reopening and its maximum raise.
    A raise below min_raise_to is allowed only at the player's own all-in cap.
    """
    hero = next(p for p in state.players if p.seat == state.hero)
    call = min(max(0, state.to_call), hero.stack)
    if action.action == "fold":
        return Action("check" if call == 0 else "fold")
    if action.action in ("check", "call"):
        return Action("call" if call else "check")
    if action.action != "raise":
        raise ValueError("unknown action")
    cap = min(state.max_raise_to, hero.committed + hero.stack)
    current = hero.committed + state.to_call
    if not state.can_raise or cap <= current or hero.stack <= call:
        return Action("call" if call else "check")
    if cap < state.min_raise_to and cap != hero.committed + hero.stack:
        return Action("call" if call else "check")
    amount = min(cap, max(state.min_raise_to, int(action.amount)))
    return Action("raise", amount)


def validate_state(s):
    """Optional boundary validation; call outside timed gameplay if possible."""
    if len(s.hole) != 2 or len(s.board) not in (0, 3, 4, 5):
        raise ValueError("invalid hole or board length")
    cards = (*s.hole, *s.board)
    if len(set(cards)) != len(cards) or any(not 0 <= c < 52 for c in cards):
        raise ValueError("invalid or repeated card")
    seats = [p.seat for p in s.players]
    if len(set(seats)) != len(seats) or s.hero not in seats or s.button not in seats:
        raise ValueError("invalid seats")
    if s.big_blind <= 0 or min(s.pot, s.to_call, s.min_raise_to, s.max_raise_to) < 0:
        raise ValueError("invalid pot or betting bounds")
    if any(min(p.stack, p.committed, p.contributed) < 0 for p in s.players):
        raise ValueError("negative player chips")
    if s.payouts and (any(x < 0 for x in s.payouts) or any(a < b for a, b in zip(s.payouts, s.payouts[1:]))):
        raise ValueError("payouts must be nonnegative and descending")
