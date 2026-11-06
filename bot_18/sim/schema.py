"""Local observation/action contract. All monetary amounts are integer chips.

Cards are 0..51: rank = card // 4 + 2; suit = card % 4.
An agent implements ``act(observation_dict) -> action_dict``. The dataclasses
document the contract; they are optional conveniences rather than dependencies
of the competition bot. No official Agent interface has been supplied.
"""
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Action:
    action: Literal["fold", "check", "call", "raise"]
    amount: int = 0  # raise: TOTAL contribution on the CURRENT street

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action, "amount": self.amount}


@dataclass(frozen=True, slots=True)
class Player:
    seat: int
    stack: int  # uncommitted chips; excludes bet and invested
    bet: int  # current street contribution; INCLUDED in invested and pot
    invested: int  # entire hand contribution; includes current street
    folded: bool
    all_in: bool

    def as_dict(self) -> dict[str, Any]:
        return {"seat": self.seat, "stack": self.stack, "bet": self.bet,
                "invested": self.invested, "folded": self.folded,
                "all_in": self.all_in}


@dataclass(frozen=True, slots=True)
class Observation:
    hand_id: int
    street: str
    hero: int
    button: int
    small_blind: int
    big_blind: int
    hole: tuple[int, int]
    board: tuple[int, ...]
    players: tuple[Player, ...]
    pot: int
    to_call: int  # capped by hero's stack, NOT by pot eligibility
    min_raise_to: int  # full minimum; may exceed max for short all-in
    max_raise_to: int  # hero.stack + hero.bet
    can_raise: bool
    history: tuple[dict[str, Any], ...]
    payouts: tuple[float, ...] | None = None
    hands_remaining: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"hand_id": self.hand_id, "street": self.street,
                "hero": self.hero, "button": self.button,
                "small_blind": self.small_blind, "big_blind": self.big_blind,
                "hole": list(self.hole), "board": list(self.board),
                "players": [p.as_dict() for p in self.players], "pot": self.pot,
                "to_call": self.to_call, "min_raise_to": self.min_raise_to,
                "max_raise_to": self.max_raise_to, "can_raise": self.can_raise,
                "history": [dict(h) for h in self.history],
                "payouts": None if self.payouts is None else list(self.payouts),
                "hands_remaining": self.hands_remaining}

