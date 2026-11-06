"""Independent local no-limit Hold'em simulator; not an official contest API.

Cards are integers 0..51: rank=card//4+2, suit=card%4. Raises use total
street commitment, including chips already bet on this street. The default
blind progression after 10/20 is a LOCAL EXPERIMENT ASSUMPTION. No official
payouts were supplied, so no score is computed unless payouts are explicit.

Betting/odd-chip conventions follow public Poker TDA 2026 rules 21, 34, 36,
45 and 49: https://www.pokertda.com/view-poker-tda-rules/ . Physical chip
declaration corrections do not apply to this explicit digital action API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
from typing import Any, Callable, Mapping, Sequence

from poker_eval import evaluate


STREETS = ("preflop", "flop", "turn", "river")
LOCAL_BLIND_SCHEDULE = (
    (10, 20), (15, 30), (25, 50), (40, 80),
    (60, 120), (100, 200), (150, 300), (250, 500),
)


def _next(seat: int, eligible: Sequence[bool]) -> int:
    for offset in range(1, len(eligible) + 1):
        candidate = (seat + offset) % len(eligible)
        if eligible[candidate]:
            return candidate
    raise ValueError("no eligible seat")


def icm_equities(stacks: Sequence[int | float], payouts: Sequence[float]) -> list[float]:
    """Exact independent chip model for strictly positive active stacks.

    Zero-stack entries receive zero; callers assign elimination awards first.
    Payouts are first through last place, with absent trailing places worth zero.
    This terminal evaluator is not used inside timed agent decisions.
    """
    if any(s < 0 for s in stacks):
        raise ValueError("negative stack")
    active = [i for i, s in enumerate(stacks) if s > 0]
    result = [0.0] * len(stacks)
    n = len(active)
    if not n:
        return result
    chips = [float(stacks[i]) for i in active]
    total = sum(chips)
    count = 1 << n
    sums = [0.0] * count
    probabilities = [0.0] * count
    probabilities[0] = 1.0
    for mask in range(1, count):
        bit = mask & -mask
        sums[mask] = sums[mask ^ bit] + chips[bit.bit_length() - 1]
    for mask in range(count - 1):
        place = mask.bit_count()
        prob = probabilities[mask]
        if not prob:
            continue
        remaining = total - sums[mask]
        payoff = float(payouts[place]) if place < len(payouts) else 0.0
        for j, seat in enumerate(active):
            bit = 1 << j
            if not mask & bit:
                p = prob * chips[j] / remaining
                probabilities[mask | bit] += p
                result[seat] += p * payoff
    return result


@dataclass
class DecisionStats:
    decisions: list[int]
    invalids: list[int]
    errors: list[int]
    overruns: list[int]
    latencies_ns: list[list[int]]

    @classmethod
    def empty(cls, n: int) -> "DecisionStats":
        return cls([0] * n, [0] * n, [0] * n, [0] * n, [[] for _ in range(n)])

    def add(self, other: "DecisionStats") -> None:
        for name in ("decisions", "invalids", "errors", "overruns"):
            a, b = getattr(self, name), getattr(other, name)
            for i in range(len(a)):
                a[i] += b[i]
        for a, b in zip(self.latencies_ns, other.latencies_ns):
            a.extend(b)


@dataclass
class HandResult:
    stacks: list[int]
    contributions: list[int]
    awards: list[int]
    refunds: list[int]
    board: list[int]
    showdown: dict[int, list[int]]
    history: list[dict[str, Any]]
    button: int
    small_blind_seat: int
    big_blind_seat: int
    stats: DecisionStats


class HandState:
    """Steppable hand engine with private state never included in observations."""

    def __init__(
        self, stacks: Sequence[int], button: int = 0, small_blind: int = 10,
        big_blind: int = 20, hand_id: int = 0, seed: int = 0,
        deck: Sequence[int] | None = None, payouts: Sequence[float] | None = None,
        small_blind_seat: int | None = None, big_blind_seat: int | None = None,
    ) -> None:
        self.n = len(stacks)
        if not 2 <= self.n <= 8 or sum(s > 0 for s in stacks) < 2:
            raise ValueError("a hand requires 2..8 seats and at least two positive stacks")
        if any(type(s) is not int or s < 0 for s in stacks):
            raise ValueError("stacks must be nonnegative integers")
        if not 0 <= button < self.n or not 0 < small_blind <= big_blind:
            raise ValueError("invalid button or blinds")
        self.stacks = list(stacks)
        self.hand_start_stacks = list(stacks)
        self.total_chips = sum(stacks)
        self.active = [s > 0 for s in stacks]
        self.folded = [not a for a in self.active]
        self.button = button
        self.small_blind, self.big_blind = small_blind, big_blind
        self.hand_id = hand_id
        self.payouts = list(payouts) if payouts is not None else None
        if sum(self.active) == 2:
            if not self.active[button]:
                raise ValueError("heads-up button must be an active seat")
            self.small_blind_seat = button
            self.big_blind_seat = _next(button, self.active)
        else:
            self.small_blind_seat = (
                _next(button, self.active) if small_blind_seat is None else small_blind_seat
            )
            self.big_blind_seat = (
                _next(self.small_blind_seat, self.active)
                if big_blind_seat is None else big_blind_seat
            )
        if not (0 <= self.small_blind_seat < self.n and 0 <= self.big_blind_seat < self.n):
            raise ValueError("invalid blind seat")
        if not self.active[self.big_blind_seat]:
            raise ValueError("big blind must be active")
        if self.small_blind_seat == self.big_blind_seat:
            raise ValueError("blinds must occupy different seats")
        self.deck = list(range(52)) if deck is None else list(deck)
        if len(self.deck) != 52 or set(self.deck) != set(range(52)):
            raise ValueError("deck must be a permutation of 0..51")
        if deck is None:
            random.Random(seed).shuffle(self.deck)
        self.deck_index = 0
        self.holes = [[] for _ in stacks]
        deal_order = self._clockwise(button, self.active)
        for _ in range(2):
            for seat in deal_order:
                self.holes[seat].append(self._draw())
        self.board: list[int] = []
        self.bets = [0] * self.n
        self.contributions = [0] * self.n
        self.awards = [0] * self.n
        self.refunds = [0] * self.n
        self.history: list[dict[str, Any]] = []
        self.street_index = 0
        self.current_bet = big_blind  # nominal bring-in even with a short blind
        self.last_full_raise = big_blind
        self.last_acted_at: list[int | None] = [None] * self.n
        self.reopen_at = [0] * self.n
        self.street_raise_count = 0
        self.last_aggressor: int | None = None
        self.pending = {i for i in range(self.n) if self.active[i]}
        self.actor: int | None = None
        self.done = False
        self.stats = DecisionStats.empty(self.n)
        self._post_blind(self.small_blind_seat, small_blind, "small")
        self._post_blind(self.big_blind_seat, big_blind, "big")
        self._advance(self.big_blind_seat)

    @property
    def street(self) -> str:
        return STREETS[self.street_index]

    @property
    def pot(self) -> int:
        return sum(self.contributions)

    def _draw(self) -> int:
        card = self.deck[self.deck_index]
        self.deck_index += 1
        return card

    def _clockwise(self, after: int, eligible: Sequence[bool]) -> list[int]:
        return [
            (after + offset) % self.n for offset in range(1, self.n + 1)
            if eligible[(after + offset) % self.n]
        ]

    def _post_blind(self, seat: int, nominal: int, kind: str) -> None:
        amount = min(nominal, self.stacks[seat])
        if not amount:
            return
        before, stack_before = self.pot, self.stacks[seat]
        self._commit(seat, amount)
        self.history.append({
            "seat": seat, "street": "preflop", "action": "blind", "blind": kind,
            "amount": amount, "total": self.bets[seat], "pot_before": before,
            "to_call_before": 0, "stack_before": stack_before,
        })

    def _commit(self, seat: int, amount: int) -> None:
        assert 0 <= amount <= self.stacks[seat]
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.contributions[seat] += amount

    def can_raise(self, seat: int) -> bool:
        if self.folded[seat] or self.stacks[seat] <= 0:
            return False
        if self.bets[seat] + self.stacks[seat] <= self.current_bet:
            return False
        if not any(i != seat and not self.folded[i] and self.stacks[i] > 0 for i in range(self.n)):
            return False
        return self.last_acted_at[seat] is None or self.current_bet >= self.reopen_at[seat]

    def observation(self) -> dict[str, Any]:
        if self.done or self.actor is None:
            raise ValueError("there is no pending decision")
        seat = self.actor
        to_call = max(0, self.current_bet - self.bets[seat])
        can_raise = self.can_raise(seat)
        legal = ["fold", "call"] if to_call else ["check"]
        if can_raise:
            legal.append("raise")
        obs = {
            "hand_id": self.hand_id, "street": self.street, "hero": seat,
            "button": self.button, "hole": self.holes[seat][:], "board": self.board[:],
            "stacks": self.stacks[:], "bets": self.bets[:],
            "contributions": self.contributions[:], "folded": self.folded[:],
            "active": self.active[:], "hand_start_stacks": self.hand_start_stacks[:],
            "pot": self.pot, "to_call": to_call,
            "min_raise_to": self.current_bet + self.last_full_raise,
            "max_raise_to": self.bets[seat] + self.stacks[seat],
            "can_raise": can_raise, "legal_actions": legal,
            "small_blind": self.small_blind, "big_blind": self.big_blind,
            "small_blind_seat": self.small_blind_seat, "big_blind_seat": self.big_blind_seat,
            "history": [event.copy() for event in self.history],
            "street_raise_count": self.street_raise_count, "last_aggressor": self.last_aggressor,
            "actor_order": self._clockwise(seat - 1, [i in self.pending for i in range(self.n)]),
        }
        if self.payouts is not None:
            obs["payouts"] = self.payouts[:]
        return obs

    def _valid_action(self, action: Any, seat: int) -> bool:
        if not isinstance(action, Mapping):
            return False
        kind = action.get("action")
        to_call = max(0, self.current_bet - self.bets[seat])
        if kind == "fold" or kind == "call":
            return to_call > 0
        if kind == "check":
            return to_call == 0
        if kind != "raise" or not self.can_raise(seat):
            return False
        amount = action.get("amount")
        if type(amount) is not int:
            return False
        maximum = self.bets[seat] + self.stacks[seat]
        return (
            self.current_bet < amount <= maximum
            and (amount >= self.current_bet + self.last_full_raise or amount == maximum)
        )

    def fallback(self) -> dict[str, Any]:
        if self.actor is None:
            raise ValueError("no actor")
        return {"action": "check" if self.current_bet == self.bets[self.actor] else "fold"}

    def apply(self, action: Mapping[str, Any], *, strict: bool = False) -> dict[str, Any]:
        """Apply one action; invalid digital actions check/fold unless strict=True."""
        if self.done or self.actor is None:
            raise ValueError("hand already complete")
        seat = self.actor
        if not self._valid_action(action, seat):
            if strict:
                raise ValueError(f"illegal action {action!r}")
            self.stats.invalids[seat] += 1
            action = self.fallback()
        kind = action["action"]
        before = self.pot
        stack_before = self.stacks[seat]
        to_call = max(0, self.current_bet - self.bets[seat])
        old_bet = self.current_bet
        amount = 0
        self.pending.discard(seat)
        if kind == "fold":
            self.folded[seat] = True
        elif kind == "call":
            amount = min(to_call, self.stacks[seat])
            self._commit(seat, amount)
        elif kind == "raise":
            target = action["amount"]
            amount = target - self.bets[seat]
            self._commit(seat, amount)
            increment = target - old_bet
            self.current_bet = target
            if increment >= self.last_full_raise:
                self.last_full_raise = increment
            self.street_raise_count += 1
            self.last_aggressor = seat
            self.pending.update(
                i for i in range(self.n)
                if i != seat and not self.folded[i] and self.stacks[i] > 0 and self.bets[i] < target
            )
        self.last_acted_at[seat] = self.current_bet
        self.reopen_at[seat] = self.current_bet + self.last_full_raise
        event = {
            "seat": seat, "street": self.street, "action": kind, "amount": amount,
            "total": self.bets[seat], "pot_before": before,
            "to_call_before": to_call, "stack_before": stack_before,
        }
        self.history.append(event)
        self._advance(seat)
        return event

    def _advance(self, previous_actor: int) -> None:
        while not self.done:
            live = [i for i in range(self.n) if not self.folded[i]]
            if len(live) == 1:
                self._refund_uncalled()
                winner = live[0]
                self.awards[winner] += self.pot
                self.stacks[winner] += self.pot
                self.done, self.actor = True, None
                self._check_conservation()
                return
            solvent = {i for i in live if self.stacks[i] > 0}
            self.pending.intersection_update(solvent)
            if len(solvent) == 1:
                sole = next(iter(solvent))
                actual_opposing_bet = max(self.bets[i] for i in live if i != sole)
                if self.bets[sole] >= actual_opposing_bet:
                    self.pending.clear()
                else:
                    self.current_bet = actual_opposing_bet
                    self.pending = {sole}
            if self.pending:
                self.actor = _next(previous_actor, [i in self.pending for i in range(self.n)])
                self._check_conservation()
                return
            self._refund_uncalled()
            if self.street_index == 3:
                self._showdown()
                return
            self.street_index += 1
            self._draw()  # burn; never exposed in observations
            self.board.extend(self._draw() for _ in range(3 if self.street_index == 1 else 1))
            self.bets = [0] * self.n
            self.current_bet = 0
            self.last_full_raise = self.big_blind
            self.last_acted_at = [None] * self.n
            self.reopen_at = [0] * self.n
            self.street_raise_count = 0
            self.last_aggressor = None
            self.pending = solvent if len(solvent) > 1 else set()
            previous_actor = self.button

    def _refund_uncalled(self) -> None:
        ordered = sorted(range(self.n), key=lambda i: self.bets[i], reverse=True)
        winner = ordered[0]
        excess = self.bets[winner] - self.bets[ordered[1]]
        if excess > 0 and not self.folded[winner]:
            before = self.pot
            self.bets[winner] -= excess
            self.contributions[winner] -= excess
            self.stacks[winner] += excess
            self.refunds[winner] += excess
            self.history.append({
                "seat": winner, "street": self.street, "action": "refund", "amount": excess,
                "total": self.bets[winner], "pot_before": before,
                "to_call_before": 0, "stack_before": self.stacks[winner] - excess,
            })

    def _showdown(self) -> None:
        ranks = {i: evaluate(self.holes[i] + self.board) for i in range(self.n) if not self.folded[i]}
        previous = 0
        for level in sorted(set(self.contributions) - {0}):
            contributors = [i for i in range(self.n) if self.contributions[i] >= level]
            amount = (level - previous) * len(contributors)
            eligible = [i for i in contributors if not self.folded[i]]
            if not eligible:
                raise AssertionError("side pot has no eligible player")
            best = max(ranks[i] for i in eligible)
            winners = [i for i in self._clockwise(self.button, self.active) if i in eligible and ranks[i] == best]
            share, odd = divmod(amount, len(winners))
            for j, seat in enumerate(winners):
                award = share + (j < odd)
                self.stacks[seat] += award
                self.awards[seat] += award
            previous = level
        self.done, self.actor = True, None
        self._check_conservation()

    def _check_conservation(self) -> None:
        if any(s < 0 for s in self.stacks):
            raise AssertionError("negative stack")
        if self.done:
            assert sum(self.stacks) == self.total_chips
            assert sum(self.awards) == self.pot
        else:
            assert sum(self.stacks) + self.pot == self.total_chips

    def result(self) -> HandResult:
        if not self.done:
            raise ValueError("hand not complete")
        live = [i for i in range(self.n) if not self.folded[i]]
        shown = {i: self.holes[i][:] for i in live} if len(live) > 1 else {}
        return HandResult(
            self.stacks[:], self.contributions[:], self.awards[:], self.refunds[:],
            self.board[:], shown, [e.copy() for e in self.history], self.button,
            self.small_blind_seat, self.big_blind_seat, self.stats,
        )

    def run(
        self, agents: Sequence[Any], enforce_time_limit: bool = False,
        time_limit_ns: int = 1_000_000,
        observer: Callable[[dict[str, Any], Any, int], None] | None = None,
    ) -> HandResult:
        if len(agents) != self.n:
            raise ValueError("one agent per seat required")
        while not self.done:
            assert self.actor is not None
            seat = self.actor
            obs = self.observation()  # building observations is outside decision timing
            self.stats.decisions[seat] += 1
            start = time.perf_counter_ns()
            try:
                action = agents[seat].act(obs)
            except Exception:
                elapsed = time.perf_counter_ns() - start
                self.stats.errors[seat] += 1
                action = self.fallback()
            else:
                elapsed = time.perf_counter_ns() - start
            self.stats.latencies_ns[seat].append(elapsed)
            if elapsed > time_limit_ns:
                self.stats.overruns[seat] += 1
                if enforce_time_limit:
                    action = self.fallback()
            if observer is not None:
                observer(obs, action, elapsed)
            self.apply(action)
        return self.result()


@dataclass
class TournamentResult:
    stacks: list[int]
    scores: list[float] | None
    ranks: list[float | None]
    hands: int
    champion: int | None
    stats: DecisionStats
    hand_results: list[HandResult] = field(default_factory=list)

    @property
    def chip_shares(self) -> list[float]:
        total = sum(self.stacks)
        return [s / total for s in self.stacks]


class Tournament:
    """Bounded local sit-and-go, with supplied payout vectors only.

    A fresh agent object per table is recommended. If an object provides a
    no-argument reset() method, it is called at table start. Seat IDs are local.
    """

    def __init__(
        self, agents: Sequence[Any], seed: int = 0, starting_stack: int = 2000,
        stacks: Sequence[int] | None = None, max_hands: int = 200,
        blind_schedule: Sequence[tuple[int, int]] = LOCAL_BLIND_SCHEDULE,
        hands_per_level: int = 25, payouts: Sequence[float] | None = None,
        enforce_time_limit: bool = False, time_limit_ns: int = 1_000_000,
        record_history: bool = False, button: int | None = None,
        observer: Callable[[dict[str, Any], Any, int], None] | None = None,
    ) -> None:
        self.agents = list(agents)
        self.n = len(agents)
        if not 2 <= self.n <= 8:
            raise ValueError("2..8 agents required")
        self.stacks = list(stacks) if stacks is not None else [starting_stack] * self.n
        if len(self.stacks) != self.n or any(type(s) is not int or s < 0 for s in self.stacks):
            raise ValueError("invalid stacks")
        if not any(self.stacks) or max_hands < 0 or hands_per_level < 1:
            raise ValueError("invalid tournament bounds")
        self.schedule = list(blind_schedule)
        if not self.schedule or any(not 0 < sb <= bb for sb, bb in self.schedule):
            raise ValueError("invalid blind schedule")
        if payouts is not None and (len(payouts) > self.n or any(p < 0 for p in payouts)):
            raise ValueError("payouts must be nonnegative and no longer than the seat count")
        if payouts is not None and any(payouts[i] < payouts[i + 1] for i in range(len(payouts) - 1)):
            raise ValueError("payouts must be first-to-last, nonincreasing")
        self.seed, self.max_hands, self.hands_per_level = seed, max_hands, hands_per_level
        self.payouts = list(payouts) if payouts is not None else None
        self.enforce_time_limit, self.time_limit_ns = enforce_time_limit, time_limit_ns
        self.record_history, self.observer = record_history, observer
        self.rng = random.Random(seed)
        alive = [s > 0 for s in self.stacks]
        self.button = self.rng.choice([i for i in range(self.n) if alive[i]]) if button is None else button
        if not 0 <= self.button < self.n or not alive[self.button]:
            raise ValueError("initial button must be active")
        self.small_blind_seat = self.button if sum(alive) == 2 else _next(self.button, alive)
        self.big_blind_seat = _next(self.small_blind_seat, alive)

    def run(self) -> TournamentResult:
        stats = DecisionStats.empty(self.n)
        histories: list[HandResult] = []
        scores = [0.0] * self.n if self.payouts is not None else None
        ranks: list[float | None] = [None] * self.n
        hands = 0
        for agent in self.agents:
            reset = getattr(agent, "reset", None)
            if reset is not None:
                reset()
        while hands < self.max_hands and sum(s > 0 for s in self.stacks) > 1:
            alive_before = sum(s > 0 for s in self.stacks)
            start_stacks = self.stacks[:]
            sb, bb = self.schedule[min(hands // self.hands_per_level, len(self.schedule) - 1)]
            hand = HandState(
                self.stacks, self.button, sb, bb, hand_id=hands,
                seed=self.rng.getrandbits(64), payouts=self.payouts,
                small_blind_seat=self.small_blind_seat, big_blind_seat=self.big_blind_seat,
            )
            result = hand.run(self.agents, self.enforce_time_limit, self.time_limit_ns, self.observer)
            self.stacks = result.stacks[:]
            stats.add(result.stats)
            if self.record_history:
                histories.append(result)
            hands += 1
            eliminated = [i for i in range(self.n) if start_stacks[i] > 0 and self.stacks[i] == 0]
            # Simultaneous eliminations: larger hand-start stacks finish higher;
            # exact ties split the affected local prizes and rank positions.
            place = alive_before
            for size in sorted({start_stacks[i] for i in eliminated}):
                tied = [i for i in eliminated if start_stacks[i] == size]
                places = list(range(place - len(tied) + 1, place + 1))
                for seat in tied:
                    ranks[seat] = sum(places) / len(places)
                    if scores is not None and self.payouts is not None:
                        scores[seat] = sum(
                            self.payouts[p - 1] if p <= len(self.payouts) else 0.0 for p in places
                        ) / len(places)
                place -= len(tied)
            alive = [s > 0 for s in self.stacks]
            if sum(alive) < 2:
                break
            next_bb = _next(self.big_blind_seat, alive)
            if sum(alive) == 2:
                self.button = _next(next_bb, alive)
                self.small_blind_seat = self.button
            else:
                self.button = self.small_blind_seat
                self.small_blind_seat = self.big_blind_seat
            self.big_blind_seat = next_bb
        survivors = [i for i, s in enumerate(self.stacks) if s > 0]
        champion = survivors[0] if len(survivors) == 1 else None
        if champion is not None:
            ranks[champion] = 1.0
        if scores is not None and self.payouts is not None:
            remaining = icm_equities(self.stacks, self.payouts[:len(survivors)])
            for seat in survivors:
                scores[seat] = remaining[seat]
        return TournamentResult(self.stacks[:], scores, ranks, hands, champion, stats, histories)
