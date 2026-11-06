"""Local interface. The competition's unpublished Agent API requires an adapter."""

from dataclasses import dataclass, field
from typing import Optional
from .cards import card


@dataclass(frozen=True, slots=True)
class Action:
    kind: str
    amount: int = 0  # raise TO total street contribution; zero for other actions

    def as_dict(self):
        return {"action": self.kind, "amount": self.amount}


@dataclass(frozen=True, slots=True)
class Player:
    seat: int
    stack: int  # remaining, excludes contributions
    street_bet: int = 0
    invested: int = 0
    folded: bool = False
    sitting_out: bool = False


@dataclass(frozen=True, slots=True)
class Event:
    seat: int
    street: int  # 0 preflop, 1 flop, 2 turn, 3 river
    kind: str
    amount: int = 0


@dataclass(frozen=True, slots=True)
class Observation:
    hole: tuple[int, int]
    board: tuple[int, ...]
    hero: int
    button: int
    players: tuple[Player, ...]
    pot: int  # all chips contributed including street bets, counted exactly once
    big_blind: int
    to_call: int  # amount to match, capped at hero stack
    min_raise_to: int
    max_raise_to: int  # hero's full street bet + remaining stack
    can_raise: bool = True
    history: tuple[Event, ...] = ()  # full current hand history, no hidden cards
    hand_id: int = 0
    table_id: str = "local"
    payouts: Optional[tuple[float, ...]] = None  # actual table points, if known
    hands_remaining: Optional[int] = None
    # An adapter should populate these from official legal-action metadata.
    legal: tuple[str, ...] = ("fold", "check", "call", "raise")

    @property
    def street(self):
        return 0 if not self.board else len(self.board) - 2

    @classmethod
    def from_dict(cls, d):
        allowed = set(cls.__dataclass_fields__)
        extra = set(d) - allowed
        if extra:
            raise ValueError("unknown observation fields: " + ", ".join(sorted(extra)))
        data = dict(d)
        data["hole"] = tuple(card(c) for c in d["hole"])
        data["board"] = tuple(card(c) for c in d.get("board", ()))
        data["players"] = tuple(Player(**p) for p in d["players"])
        data["history"] = tuple(Event(**e) for e in d.get("history", ()))
        if d.get("payouts") is not None:
            data["payouts"] = tuple(d["payouts"])
        if "legal" in d:
            data["legal"] = tuple(d["legal"])
        obs = cls(**data)
        obs.validate()
        return obs

    def validate(self):
        cards = self.hole + self.board
        if len(self.hole) != 2 or len(self.board) not in (0, 3, 4, 5):
            raise ValueError("need two hole cards and 0/3/4/5 board cards")
        if len(set(cards)) != len(cards) or any(not isinstance(c, int) or c < 0 or c >= 52 for c in cards):
            raise ValueError("invalid or duplicated cards")
        seats = [p.seat for p in self.players]
        if len(set(seats)) != len(seats) or self.hero not in seats or self.button not in seats:
            raise ValueError("seats must be unique and include hero and button")
        hero = next(p for p in self.players if p.seat == self.hero)
        if any(p.stack < 0 or p.street_bet < 0 or p.invested < p.street_bet for p in self.players):
            raise ValueError("negative chips or inconsistent contributions")
        if self.big_blind <= 0 or self.pot < 0 or not 0 <= self.to_call <= hero.stack:
            raise ValueError("invalid blind, pot, or call amount")
        if self.max_raise_to != hero.street_bet + hero.stack or self.min_raise_to < 0:
            raise ValueError("invalid raise bounds")
        if self.payouts is not None:
            if len(self.payouts) < len(self.players) or any(x < 0 for x in self.payouts):
                raise ValueError("payout vector must cover every seat and be nonnegative")
            if any(a < b for a, b in zip(self.payouts, self.payouts[1:])):
                raise ValueError("payouts must descend")


def legalize(obs, kind, amount=0):
    """The single action output gate; handles short all-ins and closed raising."""
    legal = obs.legal
    if kind == "raise" and obs.can_raise and "raise" in legal:
        hero = next(p for p in obs.players if p.seat == obs.hero)
        if obs.max_raise_to > hero.street_bet + obs.to_call:
            # min > max denotes a legal all-in below a full minimum raise.
            target = max(obs.min_raise_to, int(amount))
            return Action("raise", min(obs.max_raise_to, target))
    if kind in ("call", "raise") and obs.to_call > 0 and "call" in legal:
        return Action("call")
    if obs.to_call == 0 and "check" in legal:
        return Action("check")
    if "fold" in legal:
        return Action("fold")
    if obs.to_call > 0 and "call" in legal:
        return Action("call")
    if "check" in legal:
        return Action("check")
    raise ValueError("observation supplies no usable legal action")
