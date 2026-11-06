"""Explicit LOCAL interface, not a claim about the unpublished contest API."""

from dataclasses import dataclass, field
from typing import Optional
from math import isfinite

from .cards import card


@dataclass(frozen=True, slots=True)
class Player:
    seat: int
    stack: int
    committed: int = 0  # chips put in on this street
    invested: int = 0  # chips put in during the entire hand
    folded: bool = False
    all_in: bool = False


@dataclass(frozen=True, slots=True)
class Event:
    id: int
    seat: int
    street: str
    action: str
    amount: int = 0  # new total street commitment for bet/raise, paid chips for call
    pot_before: int = 0
    to_call: int = 0


@dataclass(frozen=True, slots=True)
class Observation:
    hand_id: int
    hero: int
    button: int
    hole: tuple[int, int]
    board: tuple[int, ...]
    players: tuple[Player, ...]
    pot: int  # includes ALL chips already committed, including this street
    to_call: int  # amount required to match; may exceed hero's stack
    min_raise_to: int  # minimum TOTAL commitment this street
    max_raise_to: int  # maximum TOTAL commitment this street
    big_blind: int
    can_raise: bool = True
    history: tuple[Event, ...] = ()  # complete current-hand public history, chronological
    payouts: tuple[float, ...] = ()  # remaining finishing-place points, descending
    table_id: str = "local"
    hands_remaining: Optional[int] = None

    @property
    def street(self):
        return ("preflop" if not self.board else "flop" if len(self.board) == 3
                else "turn" if len(self.board) == 4 else "river")

    def validate(self):
        for name in ("hand_id", "hero", "button", "pot", "to_call", "min_raise_to", "max_raise_to", "big_blind"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if len(self.hole) != 2 or len(self.board) not in (0, 3, 4, 5):
            raise ValueError("Expected two hole cards and 0/3/4/5 board cards")
        cards = self.hole + self.board
        if len(set(cards)) != len(cards) or any(type(c) is not int or not 0 <= c < 52 for c in cards):
            raise ValueError("Duplicate or invalid visible cards")
        seats = {p.seat for p in self.players}
        if len(seats) != len(self.players) or self.hero not in seats or self.button not in seats:
            raise ValueError("Players require unique seats including hero and button")
        if not 2 <= len(seats) <= 8:
            raise ValueError("Local interface supports 2--8 seated players")
        if min(self.pot, self.to_call, self.min_raise_to, self.max_raise_to) < 0 or self.big_blind <= 0:
            raise ValueError("Negative chip values or nonpositive big blind")
        for p in self.players:
            if any(type(v) is not int for v in (p.seat, p.stack, p.committed, p.invested)):
                raise ValueError("Player seats and chips must be integers")
            if min(p.stack, p.committed, p.invested) < 0 or p.committed > p.invested:
                raise ValueError("Invalid player contribution")
        if self.pot != sum(p.invested for p in self.players):
            raise ValueError("pot must equal the sum of all players' hand contributions")
        previous_id = -1
        for e in self.history:
            if (type(e.id) is not int or e.id <= previous_id or e.seat not in seats
                    or e.street not in ("preflop", "flop", "turn", "river")
                    or e.action not in ("fold", "check", "call", "raise")
                    or any(type(v) is not int or v < 0 for v in (e.amount, e.pot_before, e.to_call))):
                raise ValueError("History must contain chronological, valid public actions")
            previous_id = e.id
        h = next(p for p in self.players if p.seat == self.hero)
        if h.folded or h.all_in or h.stack <= 0:
            raise ValueError("Hero must be eligible to act")
        if self.max_raise_to > h.stack + h.committed:
            raise ValueError("Raise cap exceeds available chips")
        if self.payouts and (len(self.payouts) < len(self.players)
                             or any(not isinstance(x,(int,float)) or not isfinite(x) or x < 0 for x in self.payouts)
                             or any(a < b for a, b in zip(self.payouts, self.payouts[1:]))):
            raise ValueError("Supply descending, nonnegative remaining payouts for all players")
        return self

    @classmethod
    def from_dict(cls, d):
        values = dict(d)
        values["hole"] = tuple(card(c) for c in values["hole"])
        values["board"] = tuple(card(c) for c in values.get("board", ()))
        values["players"] = tuple(Player(**p) for p in values["players"])
        values["history"] = tuple(Event(**e) for e in values.get("history", ()))
        values["payouts"] = tuple(values.get("payouts", ()))
        return cls(**values).validate()


@dataclass(frozen=True, slots=True)
class Action:
    action: str
    amount: int = 0  # raise-to total, never an increment; 0 for fold/check/call

    def as_dict(self):
        return {"action": self.action, "amount": self.amount}


def legalize(obs, action):
    """One auditable action safety gate, including short all-in raises."""
    if action.action == "fold":
        return Action("fold" if obs.to_call else "check")
    if action.action in ("check", "call"):
        return Action("call" if obs.to_call else "check")
    if action.action != "raise" or not obs.can_raise:
        return Action("call" if obs.to_call else "check")
    h = next(p for p in obs.players if p.seat == obs.hero)
    cap = min(obs.max_raise_to, h.committed + h.stack)
    current_bet = h.committed + obs.to_call
    if cap <= current_bet:
        return Action("call" if obs.to_call else "check")
    target = min(cap, max(obs.min_raise_to, int(action.amount)))
    if target < obs.min_raise_to and target != h.committed + h.stack:
        return Action("call" if obs.to_call else "check")
    return Action("raise", target)


def contestable_pots(obs, call, villains):
    """Eligible pot before hero's call and normalized showdown pot layers.

    A short caller cannot win opponents' excess chips. A covering player can
    win a side pot while losing the main pot. Non-all-in opponents are modeled
    as continuing through all eligible layers; future calls are not added to
    the pot-odds denominator. Folded chips remain dead money in their layers.
    """
    h = next(p for p in obs.players if p.seat == obs.hero)
    limit = h.invested + call
    if limit <= 0:
        return obs.pot, None
    eligible_before = max(0, obs.pot - sum(max(0, p.invested - limit) for p in obs.players))
    if not any(p.all_in and p.invested < limit for p in villains):
        return eligible_before, None
    contributions = [min(limit, p.invested + (call if p.seat == obs.hero else 0)) for p in obs.players]
    previous = 0
    layers = {}
    for level in sorted(set(contributions) - {0}):
        amount = (level - previous) * sum(c >= level for c in contributions)
        previous = level
        contestants = tuple(i for i, p in enumerate(villains) if not p.all_in or p.invested >= level)
        layers[contestants] = layers.get(contestants, 0) + amount
    total = sum(layers.values())
    if not total or len(layers) == 1:
        return eligible_before, None
    return eligible_before, tuple((amount / total, opponents) for opponents, amount in layers.items())
