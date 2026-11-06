"""Independent local no-limit Texas Hold'em simulator.

This is a test environment, not the contest's unavailable Agent interface.  Chip
amounts are integer units.  ``raise.amount`` is the *total street commitment*,
``call`` pays as much of ``to_call`` as the player's remaining stack permits,
and the observation's pot includes all chips committed on the current street.

The local betting rules follow Poker TDA's full-raise and dead-button rules:
https://www.pokertda.com/view-poker-tda-rules/ (Rules 34, 36, 45 and 49).
Tournament blind escalation and payouts must be explicitly supplied; the
contest rules only reveal the initial blinds and escalation interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite
import random
from time import perf_counter_ns
from typing import Any, Callable, Mapping, Sequence


class IllegalAction(ValueError):
    """An agent returned an action outside the local observation's legal set."""


@dataclass
class Player:
    stack: int
    committed: int = 0
    invested: int = 0
    folded: bool = False
    all_in: bool = False


@dataclass(frozen=True)
class PotAward:
    amount: int
    eligible: tuple[int, ...]
    winners: tuple[int, ...]
    shares: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class HandResult:
    hand_id: int
    button: int
    small_blind_seat: int | None
    big_blind_seat: int
    starting_stacks: tuple[int, ...]
    ending_stacks: tuple[int, ...]
    hole_cards: tuple[tuple[int, ...], ...]
    board: tuple[int, ...]
    awards: tuple[int, ...]
    returned: tuple[int, ...]
    pots: tuple[PotAward, ...]
    history: tuple[dict[str, Any], ...]
    eliminated: tuple[tuple[int, ...], ...]
    errors: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TournamentResult:
    stacks: tuple[int, ...]
    points: tuple[float, ...]
    finishes: tuple[float, ...]
    hands: int
    decision_count: int
    invalid_actions: int
    timeouts: int
    exceptions: int
    elapsed_decision_ns: int
    observations: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def _validate_chips(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class Table:
    """Persistent table that resets supplied agents once before its first deal.

    ``agents`` passed to :meth:`play_hand` has one entry per physical seat.
    Each entry is an object exposing ``decide(observation)`` or a callable.
    Optional ``reset()`` is called once per distinct object at the first
    :meth:`play_hand`, so reusing objects across tables cannot carry table-local
    state forward. Agents retain their state between hands of the same table.
    A supplied deck is a permutation of 0..51 with its first card dealt first.
    Default table play uses a dead button, no antes, and one burn per street.

    ``strict=True`` raises on agent exceptions and illegal actions, making
    implementation bugs visible during validation.  ``strict=False`` records
    errors and checks/folds as the contest's rule 3.7 describes.  Optional timing
    enforcement measures only the call into the agent, not observation creation.
    Python cannot preempt a hung callback; timing enforcement is retrospective.
    """

    def __init__(
        self,
        stacks: Sequence[int] | None = None,
        *,
        seed: int = 0,
        small_blind: int = 10,
        big_blind: int = 20,
        button: int = 0,
        strict: bool = True,
        decision_timeout_ms: float | None = None,
        dead_button: bool = True,
        evaluator: Callable[[Sequence[int]], Any] | None = None,
        capture_observations: int = 0,
        payouts: Sequence[float] | None = None,
        max_hands: int = 200,
    ) -> None:
        initial = tuple([2000] * 8 if stacks is None else stacks)
        if not 2 <= len(initial) <= 8:
            raise ValueError("local simulator supports 2 through 8 seats")
        for i, stack in enumerate(initial):
            _validate_chips(stack, f"stack[{i}]")
        if sum(stack > 0 for stack in initial) < 2:
            raise ValueError("a new table requires at least two live players")
        if not isinstance(button, int) or not 0 <= button < len(initial):
            raise ValueError("button is a physical seat index")
        if decision_timeout_ms is not None and (
            not isfinite(decision_timeout_ms) or decision_timeout_ms <= 0
        ):
            raise ValueError("decision_timeout_ms must be positive and finite")
        if max_hands < 1 or max_hands > 200:
            raise ValueError("max_hands must be in 1..200")
        self.players = [Player(stack) for stack in initial]
        self.n_seats = len(initial)
        self.initial_total = sum(initial)
        self.initial_stacks = initial
        self.button = button
        self.small_blind_seat: int | None = None
        self.big_blind_seat = -1
        self._previous_bb: int | None = None
        self.dead_button = dead_button
        self.rng = random.Random(seed)
        self.strict = strict
        self.decision_timeout_ns = (
            None if decision_timeout_ms is None else int(decision_timeout_ms * 1_000_000)
        )
        self._evaluator = evaluator
        self.capture_observations = max(0, capture_observations)
        self.observations: list[dict[str, Any]] = []
        self.payouts = None if payouts is None else tuple(float(x) for x in payouts)
        if self.payouts is not None:
            _validate_payouts(self.payouts, len(initial))
        self.max_hands = max_hands
        self.hand_id = 0
        self._agents_initialized = False
        self.decision_count = 0
        self.invalid_actions = 0
        self.timeouts = 0
        self.exceptions = 0
        self.elapsed_decision_ns = 0
        self.set_blinds(small_blind, big_blind)
        self.history: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.board: list[int] = []
        self.hole_cards: list[tuple[int, ...]] = [()] * self.n_seats
        self.current_bet = 0
        self.last_full_raise = big_blind
        self._acted_at: list[int | None] = [None] * self.n_seats
        self.street = "preflop"
        self._hand_start = initial
        self._returned = [0] * self.n_seats

    @property
    def stacks(self) -> tuple[int, ...]:
        return tuple(player.stack for player in self.players)

    @property
    def pot(self) -> int:
        return sum(player.invested for player in self.players)

    def set_blinds(self, small_blind: int, big_blind: int) -> None:
        _validate_chips(small_blind, "small_blind")
        _validate_chips(big_blind, "big_blind")
        if big_blind < 1 or small_blind > big_blind:
            raise ValueError("require 0 <= small_blind <= big_blind and big_blind > 0")
        self.small_blind = small_blind
        self.big_blind = big_blind

    def _next_seat(self, after: int, seats: set[int]) -> int:
        for distance in range(1, self.n_seats + 1):
            seat = (after + distance) % self.n_seats
            if seat in seats:
                return seat
        raise RuntimeError("no next seat")

    def _live(self) -> set[int]:
        return {
            seat for seat, player in enumerate(self.players)
            if not player.folded and self._hand_start[seat] > 0
        }

    def _actionable(self) -> set[int]:
        return {
            seat for seat in self._live()
            if self.players[seat].stack > 0
        }

    def _positions(self, live: set[int]) -> None:
        if self._previous_bb is None:
            if len(live) == 2:
                if self.button not in live:
                    self.button = self._next_seat(self.button, live)
                self.small_blind_seat = self.button
                self.big_blind_seat = self._next_seat(self.button, live)
            else:
                if not self.dead_button and self.button not in live:
                    self.button = self._next_seat(self.button, live)
                self.small_blind_seat = self._next_seat(self.button, live)
                self.big_blind_seat = self._next_seat(self.small_blind_seat, live)
            return
        if len(live) == 2:
            # Never give an incumbent BB the BB twice when first going heads up.
            self.big_blind_seat = self._next_seat(self._previous_bb, live)
            self.button = self._next_seat(self.big_blind_seat, live)
            self.small_blind_seat = self.button
        elif self.dead_button:
            self.big_blind_seat = self._next_seat(self._previous_bb, live)
            self.small_blind_seat = self._previous_bb if self._previous_bb in live else None
            # Any vacant seat between the last live nonblind and the SB gives
            # the same action/deal/odd-chip order. Choose immediately before SB.
            self.button = (self._previous_bb - 1) % self.n_seats
        else:
            self.button = self._next_seat(self.button, live)
            self.small_blind_seat = self._next_seat(self.button, live)
            self.big_blind_seat = self._next_seat(self.small_blind_seat, live)

    def _pay(self, seat: int, amount: int) -> int:
        player = self.players[seat]
        paid = min(player.stack, amount)
        player.stack -= paid
        player.committed += paid
        player.invested += paid
        player.all_in = player.stack == 0
        return paid

    def _can_raise(self, seat: int) -> bool:
        player = self.players[seat]
        if player.committed + player.stack <= self.current_bet:
            return False
        if not any(other != seat for other in self._actionable()):
            return False
        acted = self._acted_at[seat]
        return acted is None or self.current_bet - acted >= self.last_full_raise

    def observation(self, seat: int) -> dict[str, Any]:
        """Build a fresh, private view; opponent cards and deck are never exposed."""
        player = self.players[seat]
        to_call = max(0, self.current_bet - player.committed)
        can_raise = self._can_raise(seat)
        legal = ["fold", "call" if to_call else "check"]
        if can_raise:
            legal.append("raise")
        result: dict[str, Any] = {
            "hero": seat,
            "button": self.button,
            "small_blind_seat": self.small_blind_seat,
            "big_blind_seat": self.big_blind_seat,
            "hand_id": self.hand_id,
            "street": self.street,
            "hole_cards": self.hole_cards[seat],
            "board": tuple(self.board),
            "players": [
                {
                    "stack": p.stack,
                    "committed": p.committed,
                    "invested": p.invested,
                    "folded": p.folded,
                    "all_in": p.all_in,
                    "starting_stack": self._hand_start[index],
                }
                for index, p in enumerate(self.players)
            ],
            "pot": self.pot,
            "current_bet": self.current_bet,
            "last_full_raise": self.last_full_raise,
            "to_call": to_call,
            "min_raise_to": self.current_bet + self.last_full_raise,
            "max_raise_to": player.committed + player.stack,
            "can_raise": can_raise,
            "legal_actions": tuple(legal),
            "big_blind": self.big_blind,
            "small_blind": self.small_blind,
            "history": [dict(entry) for entry in self.history],
            "starting_stack": self._hand_start[seat],
            "hand_start_stacks": self._hand_start,
            "initial_stacks": self.initial_stacks,
            "max_hands": self.max_hands,
        }
        if self.payouts is not None:
            result["payouts"] = self.payouts
        return result

    def _normalize_action(self, seat: int, action: Any) -> tuple[str, int]:
        if not isinstance(action, Mapping):
            raise IllegalAction("action must be a mapping")
        kind = action.get("action")
        player = self.players[seat]
        to_call = max(0, self.current_bet - player.committed)
        if kind == "fold":
            return "fold", 0
        if kind == "check":
            if to_call:
                raise IllegalAction("cannot check facing a bet")
            return "check", player.committed
        if kind == "call":
            # A call facing zero is a check, as in ordinary poker.
            return ("call" if to_call else "check"), min(
                self.current_bet, player.committed + player.stack
            )
        if kind != "raise":
            raise IllegalAction(f"unknown action: {kind!r}")
        target = action.get("amount")
        if isinstance(target, bool) or not isinstance(target, int):
            raise IllegalAction("raise amount must be an integer raise-to total")
        maximum = player.committed + player.stack
        if not self._can_raise(seat):
            raise IllegalAction("betting is not open for a raise")
        if target <= self.current_bet or target > maximum:
            raise IllegalAction("raise target must exceed current bet and fit stack")
        if target < self.current_bet + self.last_full_raise and target != maximum:
            raise IllegalAction("an incomplete raise must be all-in")
        return "raise", target

    def _get_action(self, agents: Sequence[Any], seat: int) -> tuple[str, int]:
        observation = self.observation(seat)
        if len(self.observations) < self.capture_observations:
            self.observations.append(observation)
        self.decision_count += 1
        start = perf_counter_ns()
        try:
            agent = agents[seat]
            action = agent.decide(observation) if hasattr(agent, "decide") else agent(observation)
        except Exception as exc:
            elapsed = perf_counter_ns() - start
            self.elapsed_decision_ns += elapsed
            self.exceptions += 1
            if self.strict:
                raise
            self.errors.append({
                "seat": seat, "street": self.street, "kind": "exception",
                "error": type(exc).__name__, "elapsed_ns": elapsed,
            })
            return self._fallback(seat)
        elapsed = perf_counter_ns() - start
        self.elapsed_decision_ns += elapsed
        if self.decision_timeout_ns is not None and elapsed > self.decision_timeout_ns:
            self.timeouts += 1
            self.errors.append({
                "seat": seat, "street": self.street, "kind": "timeout",
                "elapsed_ns": elapsed,
            })
            return self._fallback(seat)
        try:
            return self._normalize_action(seat, action)
        except IllegalAction as exc:
            self.invalid_actions += 1
            if self.strict:
                raise
            self.errors.append({
                "seat": seat, "street": self.street, "kind": "illegal_action",
                "error": str(exc), "elapsed_ns": elapsed,
            })
            return self._fallback(seat)

    def _fallback(self, seat: int) -> tuple[str, int]:
        return ("check", self.players[seat].committed) if (
            self.players[seat].committed >= self.current_bet
        ) else ("fold", 0)

    def _round(self, agents: Sequence[Any], first: int) -> None:
        pending = self._actionable()
        cursor = (first - 1) % self.n_seats
        while pending and len(self._live()) > 1:
            actionable = self._actionable()
            pending.intersection_update(actionable)
            if not pending:
                break
            if len(actionable) == 1:
                only = next(iter(actionable))
                other_committed = max(
                    (self.players[s].committed for s in self._live() if s != only),
                    default=0,
                )
                # Do not solicit meaningless checks/bets into an empty side pot.
                if self.players[only].committed >= other_committed:
                    break
            seat = self._next_seat(cursor, pending)
            cursor = seat
            player = self.players[seat]
            before = player.committed
            kind, target = self._get_action(agents, seat)
            paid = 0
            full_raise = False
            if kind == "fold":
                player.folded = True
            elif kind == "call":
                paid = self._pay(seat, target - before)
            elif kind == "raise":
                increment = target - self.current_bet
                full_raise = increment >= self.last_full_raise
                if full_raise:
                    self.last_full_raise = increment
                self.current_bet = target
                paid = self._pay(seat, target - before)
                pending.update(self._actionable() - {seat})
            self._acted_at[seat] = self.current_bet
            pending.discard(seat)
            self.history.append({
                "seat": seat,
                "street": self.street,
                "action": kind,
                "amount": target if kind == "raise" else paid,
                "paid": paid,
                "to": player.committed,
                "full_raise": full_raise,
            })
        self._return_uncalled()

    def _return_uncalled(self) -> None:
        ordered = sorted(
            ((p.committed, seat) for seat, p in enumerate(self.players)), reverse=True
        )
        highest, seat = ordered[0]
        second = ordered[1][0]
        if highest <= second:
            return
        amount = highest - second
        player = self.players[seat]
        player.committed -= amount
        player.invested -= amount
        player.stack += amount
        player.all_in = False
        self._returned[seat] += amount
        self.history.append({
            "seat": seat, "street": self.street, "action": "return",
            "amount": amount, "paid": -amount, "to": player.committed,
        })

    def _award(self) -> tuple[tuple[int, ...], tuple[PotAward, ...]]:
        live = self._live()
        awards = [0] * self.n_seats
        ranks: dict[int, Any] = {}
        if len(live) > 1:
            if len(self.board) != 5:
                raise AssertionError("contested showdown requires a five-card board")
            if self._evaluator is None:
                from pokerbot.evaluator import evaluate
                evaluator = evaluate
            else:
                evaluator = self._evaluator
            ranks = {seat: evaluator(self.hole_cards[seat] + tuple(self.board)) for seat in live}
        levels = sorted({player.invested for player in self.players if player.invested > 0})
        previous = 0
        pots: list[PotAward] = []
        for level in levels:
            contributors = [s for s, player in enumerate(self.players) if player.invested >= level]
            eligible = tuple(s for s in contributors if s in live)
            amount = (level - previous) * len(contributors)
            previous = level
            if not eligible:
                raise AssertionError("side pot has no eligible player")
            if len(eligible) == 1:
                winners = eligible
            else:
                best = max(ranks[s] for s in eligible)
                winners = tuple(s for s in eligible if ranks[s] == best)
            ordered_winners = sorted(winners, key=lambda s: (s - self.button - 1) % self.n_seats)
            quotient, remainder = divmod(amount, len(ordered_winners))
            shares = tuple(
                (seat, quotient + int(index < remainder))
                for index, seat in enumerate(ordered_winners)
            )
            for seat, share in shares:
                awards[seat] += share
            pots.append(PotAward(amount, eligible, winners, shares))
        for seat, amount in enumerate(awards):
            self.players[seat].stack += amount
        if sum(self.stacks) != self.initial_total:
            raise AssertionError("chip conservation violated")
        return tuple(awards), tuple(pots)

    def play_hand(self, agents: Sequence[Any], deck: Sequence[int] | None = None) -> HandResult:
        if len(agents) != self.n_seats:
            raise ValueError("one agent is required per physical seat")
        if self.hand_id >= self.max_hands:
            raise ValueError("table has reached its configured hand cap")
        self._hand_start = self.stacks
        active = {seat for seat, stack in enumerate(self._hand_start) if stack > 0}
        if len(active) < 2:
            raise ValueError("table already has a winner")
        self._positions(active)
        self.history = []
        self.errors = []
        self.board = []
        self.hole_cards = [()] * self.n_seats
        self._returned = [0] * self.n_seats
        for seat, player in enumerate(self.players):
            player.committed = player.invested = 0
            player.folded = seat not in active
            player.all_in = player.stack == 0
        cards = list(range(52)) if deck is None else list(deck)
        if len(cards) != 52 or set(cards) != set(range(52)):
            raise ValueError("deck must be a permutation of card IDs 0..51")
        if not self._agents_initialized:
            initialized = set()
            for agent in agents:
                identity = id(agent)
                if identity not in initialized:
                    reset = getattr(agent, "reset", None)
                    if callable(reset):
                        reset()
                    initialized.add(identity)
            self._agents_initialized = True
        if deck is None:
            self.rng.shuffle(cards)
        card_index = 0
        deal_order: list[int] = []
        cursor = self.button
        for _ in active:
            cursor = self._next_seat(cursor, active)
            deal_order.append(cursor)
        holes: list[list[int]] = [[] for _ in self.players]
        for _ in range(2):
            for seat in deal_order:
                holes[seat].append(cards[card_index])
                card_index += 1
        self.hole_cards = [tuple(hand) for hand in holes]
        self.street = "preflop"
        self.current_bet = self.big_blind
        self.last_full_raise = self.big_blind
        self._acted_at = [None] * self.n_seats
        for seat, amount, name in (
            (self.small_blind_seat, self.small_blind, "post_sb"),
            (self.big_blind_seat, self.big_blind, "post_bb"),
        ):
            if seat is not None:
                paid = self._pay(seat, amount)
                self.history.append({
                    "seat": seat, "street": "preflop", "action": name,
                    "amount": paid, "paid": paid, "to": self.players[seat].committed,
                })
        first = self.button if len(active) == 2 else self._next_seat(self.big_blind_seat, active)
        self._round(agents, first)
        for street, count in (("flop", 3), ("turn", 1), ("river", 1)):
            if len(self._live()) <= 1:
                break
            card_index += 1  # burn
            self.board.extend(cards[card_index:card_index + count])
            card_index += count
            self.street = street
            self.current_bet = 0
            self.last_full_raise = self.big_blind
            self._acted_at = [None] * self.n_seats
            for player in self.players:
                player.committed = 0
            first = self._next_seat(self.button, self._live())
            self._round(agents, first)
        awards, pots = self._award()
        eliminated = [seat for seat in active if self.players[seat].stack == 0]
        elimination_groups = tuple(
            tuple(sorted(seat for seat in eliminated if self._hand_start[seat] == stack))
            for stack in sorted({self._hand_start[seat] for seat in eliminated})
        )
        result = HandResult(
            hand_id=self.hand_id,
            button=self.button,
            small_blind_seat=self.small_blind_seat,
            big_blind_seat=self.big_blind_seat,
            starting_stacks=self._hand_start,
            ending_stacks=self.stacks,
            hole_cards=tuple(self.hole_cards),
            board=tuple(self.board),
            awards=awards,
            returned=tuple(self._returned),
            pots=pots,
            history=tuple(dict(entry) for entry in self.history),
            eliminated=elimination_groups,
            errors=tuple(dict(entry) for entry in self.errors),
        )
        self._previous_bb = self.big_blind_seat
        self.hand_id += 1
        return result


Engine = Table


def _validate_payouts(payouts: Sequence[float], n: int) -> None:
    if len(payouts) != n or any(not isfinite(value) or value < 0 for value in payouts):
        raise ValueError("payouts must have one finite nonnegative value per finishing position")
    if any(payouts[i] < payouts[i + 1] for i in range(n - 1)):
        raise ValueError("payouts must be nonincreasing from first through last")


def icm_equities(stacks: Sequence[int | float], payouts: Sequence[float]) -> tuple[float, ...]:
    """Exact independent chip model for the remaining players (at most eight).

    Only strictly positive stacks are accepted: previously eliminated seats
    receive their fixed finishing-position points before this function is used.
    Complexity is O(n^2 * 2^n), rather than enumerating n! finish orders.
    """
    chips = tuple(float(stack) for stack in stacks)
    rewards = tuple(float(payout) for payout in payouts)
    n = len(chips)
    if not 1 <= n <= 8 or any(not isfinite(x) or x <= 0 for x in chips):
        raise ValueError("ICM requires one to eight strictly positive finite stacks")
    _validate_payouts(rewards, n)

    @lru_cache(maxsize=None)
    def remaining(mask: int) -> float:
        return sum(chips[i] for i in range(n) if mask & (1 << i))

    probabilities: dict[int, float] = {(1 << n) - 1: 1.0}
    values = [0.0] * n
    for place in range(n):
        following: dict[int, float] = {}
        for mask, probability in probabilities.items():
            total = remaining(mask)
            for seat, stack in enumerate(chips):
                flag = 1 << seat
                if mask & flag:
                    finish_probability = probability * stack / total
                    values[seat] += finish_probability * rewards[place]
                    next_mask = mask ^ flag
                    if next_mask:
                        following[next_mask] = following.get(next_mask, 0.0) + finish_probability
        probabilities = following
    return tuple(values)


def run_tournament(
    agents: Sequence[Any],
    *,
    payouts: Sequence[float],
    blind_schedule: Sequence[tuple[int, int]],
    seed: int = 0,
    stacks: Sequence[int] | None = None,
    button: int = 0,
    max_hands: int = 200,
    hands_per_level: int = 25,
    strict: bool = True,
    decision_timeout_ms: float | None = None,
    capture_observations: int = 0,
    evaluator: Callable[[Sequence[int]], Any] | None = None,
) -> TournamentResult:
    """Run a *synthetic configurable* sit-and-go and allocate remaining ICM.

    No unknown official point distribution or blind levels are filled in here.
    At the hand cap surviving stacks share the unawarded top-place rewards by
    exact ICM.  Equal-stack players eliminated together split the relevant
    prizes; different starting stacks rank in descending order of those stacks.
    """
    n = len(agents)
    if not 2 <= n <= 8:
        raise ValueError("require 2 through 8 agents")
    rewards = tuple(float(x) for x in payouts)
    _validate_payouts(rewards, n)
    if not blind_schedule or hands_per_level < 1:
        raise ValueError("provide a blind schedule and positive hands_per_level")
    initial = tuple([2000] * n if stacks is None else stacks)
    if len(initial) != n or any(stack <= 0 for stack in initial):
        raise ValueError("every tournament entrant requires a positive starting stack")
    table = Table(
        initial, seed=seed, small_blind=blind_schedule[0][0], big_blind=blind_schedule[0][1],
        button=button, strict=strict, decision_timeout_ms=decision_timeout_ms,
        capture_observations=capture_observations, evaluator=evaluator,
        payouts=rewards, max_hands=max_hands,
    )
    points = [0.0] * n
    finishes = [0.0] * n
    remaining_count = n
    for hand in range(max_hands):
        small, big = blind_schedule[min(hand // hands_per_level, len(blind_schedule) - 1)]
        table.set_blinds(small, big)
        result = table.play_hand(agents)
        for group in result.eliminated:
            start = remaining_count - len(group)
            award = sum(rewards[start:remaining_count]) / len(group)
            finish = (start + 1 + remaining_count) / 2.0
            for seat in group:
                points[seat] = award
                finishes[seat] = finish
            remaining_count -= len(group)
        if remaining_count == 1:
            break
    survivors = [seat for seat, stack in enumerate(table.stacks) if stack > 0]
    remaining_values = icm_equities([table.players[seat].stack for seat in survivors], rewards[:len(survivors)])
    # Expected finishing place uses the same ICM model as remaining points.
    remaining_places = icm_equities(
        [table.players[seat].stack for seat in survivors],
        tuple(float(len(survivors) - place) for place in range(len(survivors))),
    )
    for index, seat in enumerate(survivors):
        points[seat] = remaining_values[index]
        finishes[seat] = len(survivors) + 1.0 - remaining_places[index]
    if abs(sum(points) - sum(rewards)) > 1e-7 * max(1.0, sum(rewards)):
        raise AssertionError("tournament payout conservation violated")
    return TournamentResult(
        stacks=table.stacks,
        points=tuple(points),
        finishes=tuple(finishes),
        hands=table.hand_id,
        decision_count=table.decision_count,
        invalid_actions=table.invalid_actions,
        timeouts=table.timeouts,
        exceptions=table.exceptions,
        elapsed_decision_ns=table.elapsed_decision_ns,
        observations=tuple(table.observations),
    )
