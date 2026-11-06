"""Local interface, deliberately not presented as the unknown contest interface."""
from dataclasses import dataclass
from typing import Optional
from .cards import card


@dataclass(frozen=True, slots=True)
class Player:
    seat: int
    stack: int                 # Chips behind, excluding all committed chips.
    bet: int = 0               # Contribution on this street.
    committed: int = 0         # Contribution over the whole hand.
    folded: bool = False
    all_in: bool = False


@dataclass(frozen=True, slots=True)
class Event:
    id: int                    # Monotonically increasing within a table.
    seat: int
    street: str
    action: str                # fold/check/call/raise; no blind posts here.
    amount: int = 0            # Raise-to total or call contribution.
    to_call: int = 0           # Amount faced before this action.


@dataclass(frozen=True, slots=True)
class Observation:
    hole: tuple
    board: tuple
    players: tuple
    hero: int
    button: int
    pot: int                   # All committed chips, including current bets.
    to_call: int                # Actual call cost, capped at hero's stack.
    min_raise_to: int
    max_raise_to: int
    legal_actions: tuple
    big_blind: int = 20
    hand_id: int = 0
    table_id: str = "local"
    events: tuple = ()
    payouts: tuple = ()         # Optional actual finishing-point schedule.
    eligible_pot: Optional[int] = None

    @property
    def street(self):
        return ("preflop", "", "", "flop", "turn", "river")[len(self.board)]

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        data["hole"] = tuple(card(c) for c in data["hole"])
        data["board"] = tuple(card(c) for c in data.get("board", ()))
        data["players"] = tuple(Player(**p) for p in data["players"])
        data["events"] = tuple(Event(**e) for e in data.get("events", ())[-32:])
        data["legal_actions"] = tuple(data["legal_actions"])
        data["payouts"] = tuple(data.get("payouts", ()))
        result = cls(**data)
        result.validate()
        return result

    def validate(self):
        if len(self.hole) != 2 or len(self.board) not in (0, 3, 4, 5):
            raise ValueError("Expected two hole cards and 0/3/4/5 board cards")
        cards = self.hole + self.board
        if len(set(cards)) != len(cards) or any(not isinstance(c, int) or not 0 <= c < 52 for c in cards):
            raise ValueError("Invalid or duplicate visible cards")
        seats = [p.seat for p in self.players]
        if len(set(seats)) != len(seats) or self.hero not in seats or self.button not in seats or not 2 <= len(seats) <= 8 or any(not isinstance(s, int) or not 0 <= s < 8 for s in seats):
            raise ValueError("Expected unique seats, including hero/button, at most eight")
        if any(p.stack < 0 or p.bet < 0 or p.committed < p.bet for p in self.players):
            raise ValueError("Invalid player chips")
        me = next(p for p in self.players if p.seat == self.hero)
        if self.pot < 0 or self.big_blind <= 0 or not 0 <= self.to_call <= me.stack:
            raise ValueError("Invalid pot, blind, or call cost")
        if self.pot != sum(p.committed for p in self.players):
            raise ValueError("Pot must equal the total committed by all listed seats")
        if not self.legal_actions or any(a not in ("fold", "check", "call", "raise") for a in self.legal_actions):
            raise ValueError("Invalid legal actions")
        if "raise" in self.legal_actions and not me.bet < self.max_raise_to <= me.bet + me.stack:
            raise ValueError("Invalid maximum raise-to")
        if "raise" in self.legal_actions and self.min_raise_to <= max(p.bet for p in self.players):
            raise ValueError("Minimum full raise must exceed the current bet")
        if self.payouts and (len(self.payouts) < len(self.players) or any(x < 0 for x in self.payouts)
                             or any(a < b for a, b in zip(self.payouts, self.payouts[1:]))):
            raise ValueError("Payouts must be nonnegative, descending, and cover the table")


@dataclass(frozen=True, slots=True)
class Action:
    action: str
    amount: int = 0             # Only raises carry an amount: total street bet.

    def as_dict(self):
        return {"action": self.action, "amount": self.amount}


def passive(obs):
    if "check" in obs.legal_actions:
        return Action("check")
    if "fold" in obs.legal_actions:
        return Action("fold")
    if "call" in obs.legal_actions:
        return Action("call")
    return Action("raise", obs.max_raise_to)


def legalize(obs, action):
    if action.action not in obs.legal_actions:
        return passive(obs)
    if action.action == "raise":
        # The engine, not the bot, decides whether action has reopened. If the
        # legal max is below the full minimum, this is a legal short all-in.
        lo = min(obs.min_raise_to, obs.max_raise_to)
        return Action("raise", min(obs.max_raise_to, max(lo, int(action.amount))))
    return Action(action.action)

