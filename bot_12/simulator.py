"""Original, local no-limit Hold'em test harness (not the contest interface).

Cards are rank * 4 + suit, ranks 0..12 == 2..A. Raise ``amount`` is a
player's TOTAL contribution on the current street. Pot includes current bets.
The default doubling blind schedule is synthetic; official increments/payouts
were not supplied. No rake, ante, rebuy, collusion, network, or shared state.
"""
from __future__ import annotations

from math import isfinite
import random
from time import perf_counter_ns
from typing import Any, Callable, Sequence


STREETS = ("preflop", "flop", "turn", "river")
LOCAL_BLIND_SCHEDULE = ((10, 20), (20, 40), (40, 80), (80, 160),
                        (160, 320), (320, 640), (640, 1280), (1280, 2560))


class InvalidAction(ValueError):
    pass


def icm_values(stacks: Sequence[int | float], payouts: Sequence[float]) -> list[float]:
    """Exact independent-chip-model expected points, at most eight alive seats.

    Zero-stack seats receive zero here: callers assign already-finished payouts
    separately. Payouts are top-first and may be shorter than the live field.
    """
    live = [i for i, s in enumerate(stacks) if s > 0]
    n = len(live)
    result = [0.0] * len(stacks)
    if not n:
        return result
    chips = tuple(float(stacks[i]) for i in live)
    prizes = tuple(float(payouts[k]) if k < len(payouts) else 0.0 for k in range(n))
    totals = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        bit = mask & -mask
        totals[mask] = totals[mask ^ bit] + chips[bit.bit_length() - 1]
    probability = [0.0] * (1 << n)
    probability[(1 << n) - 1] = 1.0
    for mask in range((1 << n) - 1, 0, -1):
        rank = n - mask.bit_count()
        mass = probability[mask]
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            j = bit.bit_length() - 1
            win = mass * chips[j] / totals[mask]
            result[live[j]] += win * prizes[rank]
            probability[mask ^ bit] += win
            remaining ^= bit
    return result


class HoldemHand:
    """A stepable hand. Invalid actions raise; ``play`` applies safe fallbacks.

    ``stacks`` is copied. Button must be a live seat. A supplied deck is a full
    permutation whose first element is dealt first. The dealer burns normally.
    The moving live button is a local assumption, rather than a dead button.
    """
    def __init__(self, stacks: Sequence[int], button: int, small_blind: int = 10,
                 big_blind: int = 20, *, rng: random.Random | None = None,
                 deck: Sequence[int] | None = None, hand_number: int = 0,
                 payouts: Sequence[float] | None = None):
        if not 2 <= len(stacks) <= 8:
            raise ValueError("local harness supports two through eight seats")
        if any(type(s) is not int or s < 0 for s in stacks):
            raise ValueError("stacks must be nonnegative integer chips")
        if sum(s > 0 for s in stacks) < 2 or not 0 <= button < len(stacks) or stacks[button] <= 0:
            raise ValueError("button must be live, with at least two live seats")
        if not (type(small_blind) is int and type(big_blind) is int and 0 < small_blind <= big_blind):
            raise ValueError("invalid blinds")
        self.n = len(stacks)
        self.stacks = list(stacks)
        self.starting_stacks = list(stacks)
        self.button = button
        self.small_blind, self.big_blind = small_blind, big_blind
        self.hand_number = hand_number
        self.payouts = tuple(payouts) if payouts is not None else None
        self.in_hand = [s > 0 for s in stacks]
        self.all_in = [False] * self.n
        self.contributions = [0] * self.n
        self.street_contributions = [0] * self.n
        self.holes: list[list[int]] = [[] for _ in stacks]
        self.board: list[int] = []
        self.history: list[dict[str, Any]] = []
        self.invalid_actions = [0] * self.n
        self.exceptions = [0] * self.n
        self.timeouts = [0] * self.n
        self.decisions = [0] * self.n
        self.latencies_ns: list[list[int]] = [[] for _ in stacks]
        self.done = False
        self.awards = [0] * self.n
        self.side_pots: list[dict[str, Any]] = []
        self.showdown: dict[int, int] = {}
        if deck is None:
            self.deck = list(range(52))
            (rng or random.Random()).shuffle(self.deck)
        else:
            if len(deck) != 52 or set(deck) != set(range(52)):
                raise ValueError("deck must be a permutation of all 52 cards")
            self.deck = list(deck)
        self.deck_index = 0
        order = self._clockwise(button, self.in_hand)
        for _ in range(2):
            for seat in order:
                self.holes[seat].append(self._deal())
        live = [i for i in range(self.n) if self.in_hand[i]]
        if len(live) == 2:
            self.sb_seat = button
            self.bb_seat = next(i for i in live if i != button)
            first = button
        else:
            self.sb_seat, self.bb_seat = order[:2]
            first = self._next(self.bb_seat, self.in_hand)
        self.street_index = 0
        self.current_bet = big_blind
        self.last_full_raise = big_blind
        self.acted_at: list[int | None] = [None] * self.n
        self._post_blind(self.sb_seat, small_blind, "small_blind")
        self._post_blind(self.bb_seat, big_blind, "big_blind")
        self.pending = {i for i in live if not self.all_in[i]}
        self.to_act: int | None = first
        self._progress(first, include_start=True)

    @property
    def street(self) -> str:
        return STREETS[self.street_index]

    @property
    def pot(self) -> int:
        return 0 if self.done else sum(self.contributions)

    def _deal(self) -> int:
        card = self.deck[self.deck_index]
        self.deck_index += 1
        return card

    def _clockwise(self, seat: int, flags: Sequence[bool]) -> list[int]:
        return [j for k in range(1, self.n + 1) if flags[j := (seat + k) % self.n]]

    def _next(self, seat: int, flags: Sequence[bool]) -> int:
        return self._clockwise(seat, flags)[0]

    def _pay(self, seat: int, chips: int) -> None:
        assert 0 <= chips <= self.stacks[seat]
        self.stacks[seat] -= chips
        self.contributions[seat] += chips
        self.street_contributions[seat] += chips
        self.all_in[seat] = self.stacks[seat] == 0

    def _post_blind(self, seat: int, amount: int, action: str) -> None:
        paid = min(amount, self.stacks[seat])
        self._pay(seat, paid)
        self.history.append(dict(seat=seat, street="preflop", action=action,
                                 amount=self.street_contributions[seat], paid=paid))

    def _raise_allowed(self, seat: int) -> bool:
        if not any(i != seat and self.in_hand[i] and not self.all_in[i] for i in range(self.n)):
            return False
        previous = self.acted_at[seat]
        return previous is None or self.current_bet - previous >= self.last_full_raise

    def legal_actions(self, seat: int | None = None) -> list[str]:
        seat = self.to_act if seat is None else seat
        if self.done or seat is None or seat != self.to_act:
            return []
        call = min(self.stacks[seat], max(0, self.current_bet - self.street_contributions[seat]))
        actions = ["fold", "call"] if call else ["check"]
        if self._raise_allowed(seat) and self.stacks[seat] + self.street_contributions[seat] > self.current_bet:
            actions.append("raise")
        return actions

    def observation(self, seat: int | None = None) -> dict[str, Any]:
        seat = self.to_act if seat is None else seat
        if seat is None or seat != self.to_act or self.done:
            raise ValueError("observations are available only for the acting seat")
        obs = dict(hole=list(self.holes[seat]), board=list(self.board), hero=seat,
                   button=self.button, street=self.street, street_index=self.street_index,
                   pot=self.pot, current_bet=self.current_bet,
                   min_raise_to=self.current_bet + self.last_full_raise,
                   max_raise_to=self.street_contributions[seat] + self.stacks[seat],
                   to_call=min(self.stacks[seat], max(0, self.current_bet - self.street_contributions[seat])),
                   stacks=list(self.stacks), contributions=list(self.contributions),
                   street_contributions=list(self.street_contributions),
                   in_hand=list(self.in_hand), all_in=list(self.all_in),
                   legal_actions=self.legal_actions(seat),
                   history=[event.copy() for event in self.history],
                   small_blind=self.small_blind, big_blind=self.big_blind,
                   sb_seat=self.sb_seat, bb_seat=self.bb_seat,
                   hand_number=self.hand_number,
                   active_players=sum(s > 0 for s in self.starting_stacks),
                   start_active=sum(s > 0 for s in self.starting_stacks),
                   raise_reopened=self._raise_allowed(seat))
        if self.payouts is not None:
            obs["payouts"] = self.payouts
        return obs

    def step(self, action: dict[str, Any], seat: int | None = None) -> None:
        seat = self.to_act if seat is None else seat
        if self.done or seat is None or seat != self.to_act:
            raise InvalidAction("action out of turn or after hand")
        if not isinstance(action, dict) or action.get("action") not in self.legal_actions(seat):
            raise InvalidAction("unavailable action")
        kind = action["action"]
        old_bet = self.current_bet
        paid = 0
        if kind == "fold":
            self.in_hand[seat] = False
        elif kind == "call":
            paid = min(self.stacks[seat], old_bet - self.street_contributions[seat])
            self._pay(seat, paid)
        elif kind == "raise":
            target = action.get("amount")
            maximum = self.street_contributions[seat] + self.stacks[seat]
            minimum = old_bet + self.last_full_raise
            if type(target) is not int or target <= old_bet or target > maximum:
                raise InvalidAction("raise amount must be integer street-total within stack")
            if target < minimum and target != maximum:
                raise InvalidAction("underraise is permitted only when all-in")
            paid = target - self.street_contributions[seat]
            self._pay(seat, paid)
            self.current_bet = target
            increment = target - old_bet
            if increment >= self.last_full_raise:
                self.last_full_raise = increment
            self.pending.update(i for i in range(self.n) if i != seat and self.in_hand[i]
                                and not self.all_in[i] and self.street_contributions[i] < target)
        self.acted_at[seat] = self.current_bet
        self.pending.discard(seat)
        self.history.append(dict(seat=seat, street=self.street, action=kind,
                                 amount=self.street_contributions[seat], paid=paid))
        self._progress(seat)
        self.assert_invariants()

    def _refund_uncalled(self) -> None:
        ordered = sorted(self.street_contributions, reverse=True)
        if ordered[0] > ordered[1]:
            seat = self.street_contributions.index(ordered[0])
            refund = ordered[0] - ordered[1]
            self.street_contributions[seat] -= refund
            self.contributions[seat] -= refund
            self.stacks[seat] += refund
            self.all_in[seat] = False
            self.history.append(dict(seat=seat, street=self.street, action="refund",
                                     amount=self.street_contributions[seat], paid=-refund))

    def _progress(self, after: int, include_start: bool = False) -> None:
        while not self.done:
            live = [i for i in range(self.n) if self.in_hand[i]]
            if len(live) == 1:
                self._refund_uncalled()
                winner = live[0]
                self.awards[winner] = sum(self.contributions)
                self.stacks[winner] += self.awards[winner]
                self.done = True
                self.to_act = None
                return
            capable = [i for i in live if not self.all_in[i]]
            self.pending.intersection_update(capable)
            if len(capable) == 1:
                seat = capable[0]
                # A sole player with chips can only match actual all-in wagers.
                self.current_bet = max(self.street_contributions)
                self.pending = {seat} if self.street_contributions[seat] < self.current_bet else set()
            if self.pending:
                start = after if include_start else (after + 1) % self.n
                for k in range(self.n):
                    seat = (start + k) % self.n
                    if seat in self.pending:
                        self.to_act = seat
                        return
                raise AssertionError("pending actor not found")
            self._refund_uncalled()
            if self.street_index == 3:
                self._settle_showdown()
                return
            self.street_index += 1
            self._deal()  # burn
            self.board.extend(self._deal() for _ in range(3 if self.street_index == 1 else 1))
            self.street_contributions = [0] * self.n
            self.current_bet = 0
            self.last_full_raise = self.big_blind
            self.acted_at = [None] * self.n
            capable = [i for i in live if self.stacks[i] > 0]
            self.pending = set(capable) if len(capable) >= 2 else set()
            after, include_start = self.button, False

    def _settle_showdown(self) -> None:
        from equity import evaluate
        self.showdown = {i: evaluate(self.holes[i] + self.board)
                         for i in range(self.n) if self.in_hand[i]}
        previous = 0
        for level in sorted(set(self.contributions) - {0}):
            contributors = [i for i, c in enumerate(self.contributions) if c >= level]
            amount = (level - previous) * len(contributors)
            eligible = [i for i in contributors if self.in_hand[i]]
            if not eligible:
                raise AssertionError("side pot has no eligible live hand")
            best = max(self.showdown[i] for i in eligible)
            winners = [i for i in self._clockwise(self.button, self.in_hand)
                       if i in eligible and self.showdown[i] == best]
            share, odd = divmod(amount, len(winners))
            for k, i in enumerate(winners):
                self.awards[i] += share + (k < odd)
            self.side_pots.append(dict(amount=amount, eligible=eligible, winners=winners))
            previous = level
        for i, award in enumerate(self.awards):
            self.stacks[i] += award
        self.done = True
        self.to_act = None

    def assert_invariants(self) -> None:
        assert all(type(s) is int and s >= 0 for s in self.stacks)
        assert all(c >= 0 for c in self.contributions)
        assert all(0 <= self.street_contributions[i] <= self.contributions[i] for i in range(self.n))
        assert sum(self.stacks) + (0 if self.done else sum(self.contributions)) == sum(self.starting_stacks)
        dealt = [c for hole in self.holes for c in hole] + self.board
        assert len(dealt) == len(set(dealt))
        if not self.done:
            assert self.to_act in self.pending
            assert self.in_hand[self.to_act] and not self.all_in[self.to_act]

    def play(self, bots: Sequence[Any], *, decision_limit_ns: int | None = None,
             observer: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        if len(bots) != self.n:
            raise ValueError("one bot per seat is required")
        while not self.done:
            seat = self.to_act
            assert seat is not None
            obs = self.observation()
            if observer is not None:
                observer(obs)
            start = perf_counter_ns()
            try:
                bot = bots[seat]
                action = bot.act(obs) if hasattr(bot, "act") else bot(obs)
                elapsed = perf_counter_ns() - start
            except Exception:
                elapsed = perf_counter_ns() - start
                self.exceptions[seat] += 1
                action = {"action": "check" if obs["to_call"] == 0 else "fold"}
            self.decisions[seat] += 1
            self.latencies_ns[seat].append(elapsed)
            if decision_limit_ns is not None and elapsed > decision_limit_ns:
                self.timeouts[seat] += 1
                action = {"action": "check" if obs["to_call"] == 0 else "fold"}
            try:
                self.step(action)
            except InvalidAction:
                self.invalid_actions[seat] += 1
                self.step({"action": "check" if obs["to_call"] == 0 else "fold"})
        self.assert_invariants()
        return dict(stacks=list(self.stacks), awards=list(self.awards), board=list(self.board),
                    contributions=list(self.contributions), side_pots=list(self.side_pots),
                    showdown=dict(self.showdown), history=list(self.history),
                    invalid_actions=list(self.invalid_actions), exceptions=list(self.exceptions),
                    timeouts=list(self.timeouts), decisions=list(self.decisions),
                    latencies_ns=[list(values) for values in self.latencies_ns])


def run_tournament(bots: Sequence[Any], seed: int = 0, *, starting_stack: int = 2000,
                   max_hands: int = 200, blind_schedule: Sequence[tuple[int, int]] | None = None,
                   hands_per_level: int = 25, payouts: Sequence[float] | None = None,
                   decision_limit_ns: int | None = None,
                   observer: Callable[[dict[str, Any]], None] | None = None,
                   record_hands: bool = False) -> dict[str, Any]:
    """Play one locally assumed sit-and-go; no submission/platform compatibility.

    Default payouts are synthetic (0.5, 0.3, 0.2) for local comparison only.
    Every hand's deck has a seed independent of previous bot actions, helping
    paired candidate experiments. Eliminated seats retain their original IDs.
    Simultaneous eliminations rank by starting stack; equal stacks split points.
    """
    n = len(bots)
    if not 2 <= n <= 8 or type(starting_stack) is not int or starting_stack <= 0:
        raise ValueError("two through eight bots with positive integer stacks required")
    if max_hands < 0 or hands_per_level <= 0:
        raise ValueError("invalid hand count or level length")
    schedule = tuple(blind_schedule or LOCAL_BLIND_SCHEDULE)
    if not schedule:
        raise ValueError("a nonempty blind schedule is required")
    prizes = tuple(payouts) if payouts is not None else ((0.7, 0.3) if n == 2 else (0.5, 0.3, 0.2))
    if len(prizes) > n or any(not isfinite(float(x)) or float(x) < 0 for x in prizes):
        raise ValueError("invalid local payouts")
    prizes = prizes + (0.0,) * (n - len(prizes))
    if any(prizes[i] < prizes[i + 1] for i in range(n - 1)):
        raise ValueError("payouts must be descending")
    for bot in bots:
        if hasattr(bot, "reset"):
            bot.reset()
    stacks = [starting_stack] * n
    points = [0.0] * n
    placements: list[float | None] = [None] * n
    invalid, exceptions, timeouts, decisions = ([0] * n for _ in range(4))
    latencies: list[list[int]] = [[] for _ in bots]
    eliminated: list[list[int]] = []
    logs = []
    button = seed % n
    hands = 0
    for hand_number in range(max_hands):
        alive_before = [i for i, s in enumerate(stacks) if s > 0]
        if len(alive_before) <= 1:
            break
        if stacks[button] <= 0:
            button = next((button + k) % n for k in range(1, n + 1) if stacks[(button + k) % n] > 0)
        sb, bb = schedule[min(hand_number // hands_per_level, len(schedule) - 1)]
        deck_rng = random.Random((int(seed) << 32) ^ (hand_number * 0x9E3779B97F4A7C15) ^ 0xB012)
        hand = HoldemHand(stacks, button, sb, bb, rng=deck_rng,
                          hand_number=hand_number, payouts=prizes)
        result = hand.play(bots, decision_limit_ns=decision_limit_ns, observer=observer)
        stacks = result["stacks"]
        hands += 1
        for i in range(n):
            invalid[i] += result["invalid_actions"][i]
            exceptions[i] += result["exceptions"][i]
            timeouts[i] += result["timeouts"][i]
            decisions[i] += result["decisions"][i]
            latencies[i].extend(result["latencies_ns"][i])
        busted = [i for i in alive_before if stacks[i] == 0]
        # Smallest pre-hand stack finishes lowest when several players bust.
        remain_rank = len(alive_before)
        for chips in sorted(set(hand.starting_stacks[i] for i in busted)):
            group = [i for i in busted if hand.starting_stacks[i] == chips]
            ranks = list(range(remain_rank - len(group) + 1, remain_rank + 1))
            value = sum(prizes[r - 1] for r in ranks) / len(group)
            placement = sum(ranks) / len(group)
            for i in group:
                points[i], placements[i] = value, placement
            remain_rank -= len(group)
            eliminated.append(group)
        if record_hands:
            logs.append(result)
        alive_after = [i for i, s in enumerate(stacks) if s > 0]
        if len(alive_after) <= 1:
            break
        if len(alive_after) == 2 and len(alive_before) > 2:
            # On the transition to heads-up, advance the BB so the previous BB
            # does not pay it twice in a row (TDA heads-up-button procedure).
            next_bb = next((hand.bb_seat + k) % n for k in range(1, n + 1)
                           if stacks[(hand.bb_seat + k) % n] > 0)
            button = next(i for i in alive_after if i != next_bb)
        else:
            button = next((button + k) % n for k in range(1, n + 1) if stacks[(button + k) % n] > 0)
    alive = [i for i, s in enumerate(stacks) if s > 0]
    remaining_points = icm_values(stacks, prizes[:len(alive)])
    expected_placements = icm_values(stacks, tuple(float(k) for k in range(1, len(alive) + 1)))
    for i in alive:
        points[i] = remaining_points[i]
        placements[i] = expected_placements[i]
    assert sum(stacks) == n * starting_stack
    assert abs(sum(points) - sum(prizes)) < 1e-9
    return dict(stacks=stacks, chips=stacks, points=points, placements=placements,
                hands=hands, capped=len(alive) > 1, eliminated=eliminated,
                invalid_actions=invalid, exceptions=exceptions, timeouts=timeouts,
                decisions=decisions, latencies_ns=latencies, hand_logs=logs,
                assumptions=dict(blind_schedule=list(schedule), payouts=list(prizes),
                                 hands_per_level=hands_per_level, moving_live_button=True))
