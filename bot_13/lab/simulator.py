"""Independent, offline, standard no-limit Texas Hold'em test environment.

The observation/action contract here is local, not an official Agent interface.
Cards are ``4 * rank + suit`` with ranks 2..A and suits c,d,h,s. Raises name
the new total street contribution (raise-to). ``pot`` includes all committed
chips, including current street bets. ``to_call`` is capped by hero's stack.
No antes or rake. A caller must explicitly supply later blind levels and any
placement payouts; neither is available in RULES.txt. One level lasts 25 hands.
Tournament blind progression uses a dead button; its seat may be eliminated.
The big blind advances to the next surviving seat, and a busted prior big
blind can create a dead small blind. Heads-up always gives the button the SB.

Timing records only policy.act(observation), not simulator bookkeeping. The
optional 1 ms enforcement reproduces the rules' check/fold timeout behavior.
This simulator is a validation and comparison tool, not evidence of official
compatibility or a complete game-theoretic poker solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
import random
from time import perf_counter_ns
from typing import Any, Callable, Iterable, Mapping, Sequence


STREETS = ("preflop", "flop", "turn", "river")


def _score_five(cards: Sequence[int]) -> tuple[int, ...]:
    ranks = [c // 4 + 2 for c in cards]
    counts = {r: ranks.count(r) for r in set(ranks)}
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    unique = sorted(counts, reverse=True)
    straight = 0
    if len(unique) == 5:
        if unique[0] - unique[-1] == 4:
            straight = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight = 5
    flush = len({c % 4 for c in cards}) == 1
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    if groups[0][0] == 2 and groups[1][0] == 2:
        return (2, max(groups[0][1], groups[1][1]), min(groups[0][1], groups[1][1]), groups[2][1])
    if groups[0][0] == 2:
        return (1, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    return (0, *sorted(ranks, reverse=True))


def reference_evaluate(cards: Sequence[int]) -> tuple[int, ...]:
    """Deliberately simple independent evaluator, useful for correctness checks."""
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards) or any(not 0 <= c < 52 for c in cards):
        raise ValueError("expected 5..7 distinct integer cards in 0..51")
    return max(_score_five(combo) for combo in combinations(cards, 5))


_native_evaluate: Callable | None = None
_native_attempted = False


def evaluate(cards: Sequence[int]) -> Any:
    global _native_evaluate, _native_attempted
    if not _native_attempted:
        _native_attempted = True
        try:
            from bot13.native import NativeEvaluator, native_available
            if native_available:
                _native_evaluate = NativeEvaluator().evaluate
        except (ImportError, OSError, AttributeError):
            pass
    if _native_evaluate is not None:
        return _native_evaluate(cards)
    return reference_evaluate(cards)


def icm(stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """Exact independent chip model via subset dynamic programming (n <= 8).

    Use only caller-supplied payout values. Zero-stack entries participate in
    any unresolved trailing ranks uniformly; ordinary callers pass survivors.
    """
    n = len(stacks)
    if not 1 <= n <= 8 or len(payouts) < n:
        raise ValueError("ICM needs 1..8 stacks and one payout per remaining rank")
    if any(not math.isfinite(s) or s < 0 for s in stacks):
        raise ValueError("invalid stack")
    if any(not math.isfinite(float(p)) for p in payouts[:n]):
        raise ValueError("invalid payout")
    size = 1 << n
    totals = [0.0] * size
    for mask in range(1, size):
        low = mask & -mask
        totals[mask] = totals[mask ^ low] + float(stacks[low.bit_length() - 1])
    total = totals[-1]
    prob = [0.0] * size
    prob[0] = 1.0
    result = [0.0] * n
    for mask in range(size - 1):
        rank = mask.bit_count()
        remaining = total - totals[mask]
        for i, stack in enumerate(stacks):
            if mask & (1 << i):
                continue
            conditional = float(stack) / remaining if remaining > 0 else 1.0 / (n - rank)
            p = prob[mask] * conditional
            result[i] += p * float(payouts[rank])
            prob[mask | (1 << i)] += p
    return result


def advance_blinds(stacks: Sequence[int], previous_sb_anchor: int, previous_bb: int) -> tuple[int, int | None, int, int]:
    """Return (button, actual SB or None, BB, nominal SB anchor) for next hand.

    A dead SB still has an anchor so that the button can occupy that position
    on the following hand. Keeping BB progression authoritative avoids either
    skipping a surviving player's BB or charging it consecutively.
    """
    n = len(stacks)
    alive = [i for i, stack in enumerate(stacks) if stack > 0]
    if len(alive) < 2:
        raise ValueError("blind progression requires at least two survivors")
    bb = next((previous_bb + step) % n for step in range(1, n + 1) if stacks[(previous_bb + step) % n] > 0)
    if len(alive) == 2:
        button = next(i for i in alive if i != bb)
        return button, button, bb, button
    sb = previous_bb if stacks[previous_bb] > 0 else None
    return previous_sb_anchor, sb, bb, previous_bb


@dataclass
class Player:
    seat: int
    stack: int
    hole: list[int] = field(default_factory=list)
    bet: int = 0
    committed: int = 0
    folded: bool = False
    all_in: bool = False


@dataclass
class DecisionRecord:
    seat: int
    street: str
    elapsed_ns: int
    fallback: str | None = None
    error: str | None = None


@dataclass
class HandResult:
    stacks: list[int]
    board: list[int]
    holes: list[list[int]]
    history: list[dict]
    pots: list[dict]
    refunds: list[dict]
    decisions: list[DecisionRecord]


class Hand:
    """One hand. Policy instances need only an ``act(dict) -> dict`` method."""

    def __init__(
        self,
        stacks: Sequence[int],
        agents: Sequence[Any],
        button: int,
        small_blind: int,
        big_blind: int,
        *,
        rng: random.Random | None = None,
        hand_id: int = 0,
        payouts: Sequence[float] | None = None,
        hand_limit: int = 200,
        enforce_deadline: bool = False,
        deadline_ns: int = 1_000_000,
        deck: Sequence[int] | None = None,
        blind_seats: tuple[int | None, int] | None = None,
        audit: bool = True,
    ):
        if not 2 <= len(stacks) <= 8 or len(agents) != len(stacks):
            raise ValueError("a hand needs 2..8 seats and one policy per seat")
        if any(type(s) is not int or s < 0 for s in stacks) or sum(s > 0 for s in stacks) < 2:
            raise ValueError("need nonnegative integer stacks and at least two funded seats")
        if not 0 <= button < len(stacks) or (sum(s > 0 for s in stacks) == 2 and stacks[button] <= 0):
            raise ValueError("invalid button, or dead button heads-up")
        if type(small_blind) is not int or type(big_blind) is not int or not 0 < small_blind <= big_blind:
            raise ValueError("invalid blinds")
        self.players = [Player(i, s, folded=s == 0, all_in=s == 0) for i, s in enumerate(stacks)]
        self.agents = list(agents)
        self.n = len(stacks)
        self.button = button
        self.sb = small_blind
        self.bb = big_blind
        self.hand_id = hand_id
        self.hand_limit = hand_limit
        self.payouts = list(payouts) if payouts is not None else None
        self.enforce_deadline = enforce_deadline
        self.deadline_ns = deadline_ns
        self.audit = audit
        self.initial_total = sum(stacks)
        self.board: list[int] = []
        self.history: list[dict] = []
        self.pots: list[dict] = []
        self.refunds: list[dict] = []
        self.decisions: list[DecisionRecord] = []
        self.street = "preflop"
        self.current_bet = big_blind
        self.last_full_raise = big_blind
        self.last_action_level: list[int | None] = [None] * self.n
        self.pending: set[int] = set()
        self.rng = rng or random.Random()
        if deck is None:
            self.deck = list(range(52))
            self.rng.shuffle(self.deck)
        else:
            if len(deck) != 52 or len(set(deck)) != 52 or any(type(c) is not int or not 0 <= c < 52 for c in deck):
                raise ValueError("rigged deck must be a permutation of all 52 cards")
            self.deck = list(deck)
        self.draw_index = 0
        first = self._next_live(button)
        order = self._live_order(first)
        for _ in range(2):
            for i in order:
                self.players[i].hole.append(self._draw())
        alive = len(order)
        if blind_seats is None:
            self.sb_seat = button if alive == 2 else first
            self.bb_seat = self._next_live(self.sb_seat)
        else:
            self.sb_seat, self.bb_seat = blind_seats
            if not 0 <= self.bb_seat < self.n or stacks[self.bb_seat] <= 0:
                raise ValueError("BB must be a funded seat")
            if self.sb_seat is not None and (not 0 <= self.sb_seat < self.n or stacks[self.sb_seat] <= 0 or self.sb_seat == self.bb_seat):
                raise ValueError("SB must be a funded seat distinct from BB, or None")
            if alive == 2 and (self.sb_seat != button or self.bb_seat != self._next_live(button)):
                raise ValueError("heads-up button must be the SB")
        if self.sb_seat is not None:
            self._commit(self.sb_seat, min(self.sb, self.players[self.sb_seat].stack))
        self._commit(self.bb_seat, min(self.bb, self.players[self.bb_seat].stack))

    def _draw(self) -> int:
        card = self.deck[self.draw_index]
        self.draw_index += 1
        return card

    def _next_live(self, after: int) -> int:
        for k in range(1, self.n + 1):
            i = (after + k) % self.n
            if not self.players[i].folded:
                return i
        raise RuntimeError("no live seat")

    def _live_order(self, start: int) -> list[int]:
        return [(start + k) % self.n for k in range(self.n) if not self.players[(start + k) % self.n].folded]

    def _commit(self, seat: int, amount: int) -> None:
        p = self.players[seat]
        if not 0 <= amount <= p.stack:
            raise AssertionError("invalid chip commitment")
        p.stack -= amount
        p.bet += amount
        p.committed += amount
        p.all_in = p.stack == 0
        self._check_chips()

    def _check_chips(self) -> None:
        if self.audit:
            assert all(p.stack >= 0 and p.bet >= 0 and p.committed >= p.bet for p in self.players)
            assert sum(p.stack + p.committed for p in self.players) == self.initial_total

    def _actionable(self) -> list[int]:
        return [p.seat for p in self.players if not p.folded and p.stack > 0]

    def _live(self) -> list[int]:
        return [p.seat for p in self.players if not p.folded]

    def _can_raise(self, seat: int) -> bool:
        p = self.players[seat]
        if p.folded or p.stack + p.bet <= self.current_bet:
            return False
        if not any(q.seat != seat and not q.folded and q.stack > 0 for q in self.players):
            return False
        last = self.last_action_level[seat]
        return last is None or self.current_bet - last >= self.last_full_raise

    def observation(self, seat: int) -> dict:
        p = self.players[seat]
        result = {
            "hand_id": self.hand_id,
            "street": self.street,
            "hero": seat,
            "seat_count": self.n,
            "button": self.button,
            "small_blind": self.sb,
            "big_blind": self.bb,
            "small_blind_seat": self.sb_seat,
            "big_blind_seat": self.bb_seat,
            "hole_cards": list(p.hole),
            "board": list(self.board),
            "pot": sum(q.committed for q in self.players),
            "to_call": min(p.stack, max(0, self.current_bet - p.bet)),
            "current_bet": self.current_bet,
            "min_raise_to": self.current_bet + self.last_full_raise,
            "max_raise_to": p.bet + p.stack,
            "can_raise": self._can_raise(seat),
            "players": [
                {"seat": q.seat, "stack": q.stack, "bet": q.bet, "committed": q.committed,
                 "folded": q.folded, "all_in": q.all_in}
                for q in self.players
            ],
            "history": [dict(event) for event in self.history],
            "hand_limit": self.hand_limit,
        }
        if self.payouts is not None:
            result["payouts"] = list(self.payouts)
        return result

    def _fallback(self, seat: int) -> dict:
        p = self.players[seat]
        return {"action": "check" if self.current_bet <= p.bet else "fold"}

    def _validate(self, seat: int, decision: Any) -> tuple[bool, str | None]:
        if not isinstance(decision, Mapping):
            return False, "decision is not a mapping"
        action = decision.get("action")
        p = self.players[seat]
        if action == "fold":
            return True, None
        if action == "check":
            return (self.current_bet <= p.bet, "check facing a bet" if self.current_bet > p.bet else None)
        if action == "call":
            return (self.current_bet > p.bet, "call without a bet" if self.current_bet <= p.bet else None)
        if action == "raise":
            target = decision.get("amount")
            if type(target) is not int:
                return False, "raise-to amount must be an integer"
            if not self._can_raise(seat):
                return False, "raising is not reopened or no opponent can respond"
            if target <= self.current_bet or target > p.bet + p.stack:
                return False, "raise-to outside stack/current-bet bounds"
            if target < self.current_bet + self.last_full_raise and target != p.bet + p.stack:
                return False, "subminimum raise is not all-in"
            return True, None
        return False, "unknown action"

    def _decide(self, seat: int) -> dict:
        obs = self.observation(seat)
        start = perf_counter_ns()
        error = None
        reason = None
        try:
            decision = self.agents[seat].act(obs)
        except Exception as exc:
            decision = self._fallback(seat)
            reason = "exception"
            error = f"{type(exc).__name__}: {exc}"
        elapsed = perf_counter_ns() - start
        if self.enforce_deadline and elapsed > self.deadline_ns:
            decision = self._fallback(seat)
            reason = "deadline"
        valid, detail = self._validate(seat, decision)
        if not valid:
            decision = self._fallback(seat)
            reason = "invalid"
            error = detail
        self.decisions.append(DecisionRecord(seat, self.street, elapsed, reason, error))
        return dict(decision)

    def _apply(self, seat: int, decision: Mapping) -> None:
        """Apply a validated action and maintain full/short raise reopening."""
        p = self.players[seat]
        action = decision["action"]
        prior_bet = self.current_bet
        prior_pot = sum(q.committed for q in self.players)
        prior_stack = p.stack
        prior_call = min(p.stack, max(0, prior_bet - p.bet))
        can_raise_before = self._can_raise(seat)
        paid = 0
        full_raise = False
        self.pending.discard(seat)
        if action == "fold":
            p.folded = True
        elif action == "call":
            paid = prior_call
            self._commit(seat, paid)
        elif action == "raise":
            target = decision["amount"]
            increment = target - self.current_bet
            paid = target - p.bet
            self._commit(seat, paid)
            self.current_bet = target
            full_raise = increment >= self.last_full_raise
            if full_raise:
                self.last_full_raise = increment
                self.pending = {q.seat for q in self.players if q.seat != seat and not q.folded and q.stack > 0}
            else:
                self.pending.update(q.seat for q in self.players if q.seat != seat and not q.folded and q.stack > 0 and q.bet < target)
        self.last_action_level[seat] = self.current_bet
        self.history.append({
            "seat": seat, "street": self.street, "action": action, "amount": p.bet,
            "paid": paid, "pot_before": prior_pot, "to_call": prior_call,
            "amount_paid": paid, "to_call_before": prior_call,
            "stack_before": prior_stack, "is_all_in": p.all_in,
            "current_bet_before": prior_bet, "full_raise": full_raise,
            "can_raise_before": can_raise_before,
        })
        self._check_chips()

    def _refund_uncalled(self) -> None:
        ordered = sorted(((p.bet, p.seat) for p in self.players), reverse=True)
        largest, seat = ordered[0]
        second = ordered[1][0]
        if largest > second:
            amount = largest - second
            p = self.players[seat]
            p.bet -= amount
            p.committed -= amount
            p.stack += amount
            p.all_in = p.stack == 0
            self.refunds.append({"seat": seat, "street": self.street, "amount": amount})
            self._check_chips()

    def _betting_round(self, first: int) -> None:
        self.pending = set(self._actionable())
        if len(self.pending) == 1:
            only = next(iter(self.pending))
            # A nominal short BB need not be completed into an uncontested
            # side pot when every opponent is already all-in.
            other_bet = max((p.bet for p in self.players if p.seat != only and not p.folded), default=0)
            self.current_bet = min(self.current_bet, other_bet)
            if self.players[only].bet >= self.current_bet:
                self.pending.clear()
        cursor = first
        actions = 0
        while self.pending and len(self._live()) > 1:
            eligible = [i for i in self.pending if not self.players[i].folded and self.players[i].stack > 0]
            self.pending.intersection_update(eligible)
            if not self.pending:
                break
            for step in range(self.n):
                seat = (cursor + step) % self.n
                if seat in self.pending:
                    break
            decision = self._decide(seat)
            self._apply(seat, decision)
            cursor = (seat + 1) % self.n
            actions += 1
            if actions > 100_000:
                raise RuntimeError("betting failed to terminate")
            actionable = self._actionable()
            if len(actionable) == 1:
                only = actionable[0]
                other_bet = max((p.bet for p in self.players if p.seat != only and not p.folded), default=0)
                self.current_bet = min(self.current_bet, other_bet)
                if self.players[only].bet >= self.current_bet:
                    self.pending.clear()
        self._refund_uncalled()

    def _award_uncontested(self, seat: int) -> None:
        amount = sum(p.committed for p in self.players)
        self.players[seat].stack += amount
        for p in self.players:
            p.committed = 0
            p.bet = 0
        self.pots.append({"amount": amount, "eligible": [seat], "winners": [seat], "shares": {seat: amount}})
        self._check_chips()

    def _settle_showdown(self) -> None:
        scores = {p.seat: evaluate(p.hole + self.board) for p in self.players if not p.folded}
        levels = sorted({p.committed for p in self.players if p.committed > 0})
        previous = 0
        awards = [0] * self.n
        for level in levels:
            contributors = [p.seat for p in self.players if p.committed >= level]
            amount = (level - previous) * len(contributors)
            eligible = [i for i in contributors if not self.players[i].folded]
            if not eligible:
                raise AssertionError("side pot has no eligible player")
            best = max(scores[i] for i in eligible)
            winners = [i for i in eligible if scores[i] == best]
            quotient, odd = divmod(amount, len(winners))
            order = sorted(winners, key=lambda i: (i - self.button - 1) % self.n)
            shares = {i: quotient + (k < odd) for k, i in enumerate(order)}
            for i, chips in shares.items():
                awards[i] += chips
            self.pots.append({"amount": amount, "eligible": eligible, "winners": winners, "shares": shares})
            previous = level
        for p in self.players:
            p.stack += awards[p.seat]
            p.committed = 0
            p.bet = 0
        self._check_chips()

    def run(self) -> HandResult:
        for street_no, street in enumerate(STREETS):
            self.street = street
            if street_no:
                for p in self.players:
                    p.bet = 0
                self.current_bet = 0
                self.last_full_raise = self.bb
                self.last_action_level = [None] * self.n
                self._draw()  # Burn card; its value is never revealed to policies.
                self.board.extend(self._draw() for _ in range(3 if street == "flop" else 1))
            first = (self.bb_seat + 1) % self.n if street_no == 0 else (self.button + 1) % self.n
            self._betting_round(first)
            live = self._live()
            if len(live) == 1:
                self._award_uncontested(live[0])
                break
        else:
            self._settle_showdown()
        self._check_chips()
        return HandResult(
            [p.stack for p in self.players], list(self.board), [list(p.hole) for p in self.players],
            list(self.history), list(self.pots), list(self.refunds), list(self.decisions),
        )


@dataclass
class TournamentResult:
    stacks: list[int]
    points: list[float] | None
    hands: int
    winner: int | None
    eliminations: list[dict]
    decisions: list[DecisionRecord]
    hand_results: list[HandResult]


class Tournament:
    """One table with explicit local blind schedule and optional supplied payouts."""

    def __init__(
        self,
        agents: Sequence[Any],
        *,
        blind_schedule: Sequence[tuple[int, int]],
        stacks: int | Sequence[int] = 2000,
        payouts: Sequence[float] | None = None,
        hands_per_level: int = 25,
        hand_limit: int = 200,
        seed: int = 1,
        button: int = 0,
        enforce_deadline: bool = False,
        keep_hands: bool = False,
        audit: bool = True,
    ):
        self.agents = list(agents)
        n = len(agents)
        if not 2 <= n <= 8:
            raise ValueError("table needs 2..8 policies")
        self.stacks = [stacks] * n if type(stacks) is int else list(stacks)
        if len(self.stacks) != n or any(type(s) is not int or s <= 0 for s in self.stacks):
            raise ValueError("starting stacks must be positive integers")
        self.schedule = [tuple(level) for level in blind_schedule]
        if not self.schedule or any(len(level) != 2 or any(type(v) is not int for v in level) or not 0 < level[0] <= level[1] for level in self.schedule):
            raise ValueError("must provide valid blind levels")
        if payouts is not None:
            if len(payouts) != n or any(not math.isfinite(float(p)) for p in payouts):
                raise ValueError("need one finite payout per seat")
            if any(payouts[i] < payouts[i + 1] for i in range(n - 1)):
                raise ValueError("payouts must be ordered first to last")
        if not 1 <= hands_per_level or not 1 <= hand_limit <= 200 or not 0 <= button < n:
            raise ValueError("invalid hand/level/button configuration")
        self.payouts = list(payouts) if payouts is not None else None
        self.hands_per_level = hands_per_level
        self.hand_limit = hand_limit
        self.rng = random.Random(seed)
        self.button = button
        self.enforce_deadline = enforce_deadline
        self.keep_hands = keep_hands
        self.audit = audit
        self.blind_seats: tuple[int | None, int] | None = None
        self.sb_anchor: int | None = None

    def run(self) -> TournamentResult:
        points = [0.0] * len(self.agents) if self.payouts is not None else None
        eliminations = []
        records = []
        hands = []
        completed = 0
        for hand_id in range(self.hand_limit):
            alive_before = [i for i, s in enumerate(self.stacks) if s > 0]
            if len(alive_before) == 1:
                break
            pre = list(self.stacks)
            level = self.schedule[min(hand_id // self.hands_per_level, len(self.schedule) - 1)]
            hand_state = Hand(
                self.stacks, self.agents, self.button, *level, rng=self.rng, hand_id=hand_id,
                payouts=self.payouts, hand_limit=self.hand_limit, enforce_deadline=self.enforce_deadline,
                blind_seats=self.blind_seats, audit=self.audit,
            )
            if self.sb_anchor is None:
                self.sb_anchor = hand_state.sb_seat
            hand = hand_state.run()
            self.stacks = list(hand.stacks)
            records.extend(hand.decisions)
            if self.keep_hands:
                hands.append(hand)
            completed += 1
            busted = sorted((i for i in alive_before if self.stacks[i] == 0), key=lambda i: (pre[i], i))
            occupied = len(alive_before)
            while busted:
                minimum = pre[busted[0]]
                tied = [i for i in busted if pre[i] == minimum]
                ranks = list(range(occupied - len(tied) + 1, occupied + 1))
                award = sum(self.payouts[r - 1] for r in ranks) / len(tied) if self.payouts is not None else None
                for i in tied:
                    if points is not None:
                        points[i] = award
                    eliminations.append({"seat": i, "hand_id": hand_id, "ranks": ranks, "starting_stack": pre[i], "points": award})
                    busted.remove(i)
                occupied -= len(tied)
            alive = [i for i, s in enumerate(self.stacks) if s > 0]
            if len(alive) <= 1:
                break
            self.button, next_sb, next_bb, self.sb_anchor = advance_blinds(self.stacks, self.sb_anchor, hand_state.bb_seat)
            self.blind_seats = (next_sb, next_bb)
        alive = [i for i, s in enumerate(self.stacks) if s > 0]
        if points is not None:
            remaining_points = icm([self.stacks[i] for i in alive], self.payouts[:len(alive)])
            for i, value in zip(alive, remaining_points):
                points[i] = value
            if self.audit:
                assert abs(sum(points) - sum(self.payouts)) < 1e-8 * max(1, sum(abs(p) for p in self.payouts))
        return TournamentResult(list(self.stacks), points, completed, alive[0] if len(alive) == 1 else None, eliminations, records, hands)
