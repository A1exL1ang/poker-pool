"""Independent local NLHE test engine; not an official contest simulator.

Implements no-limit raises, cumulative short-all-in reopening, side pots,
heads-up blinds, split pots, and chip conservation. Synthetic blind schedules
and payout schedules are always explicit at the command-line boundary.
"""
from dataclasses import dataclass
import random
import time
from bot7.evaluator import evaluate
from bot7.interface import Action, Event, Observation, Player


@dataclass
class HandResult:
    stacks: list
    events: list
    board: list
    contributions: list
    folded: list
    decisions: int


def settle(contributions, folded, scores, button):
    """Return per-seat pot awards, including uncalled contributions."""
    n = len(contributions)
    awards = [0] * n
    levels = sorted(set(x for x in contributions if x > 0))
    previous = 0
    for level in levels:
        contributors = [i for i, x in enumerate(contributions) if x >= level]
        amount = (level - previous) * len(contributors)
        eligible = [i for i in contributors if not folded[i]]
        if not eligible:
            raise AssertionError("Side pot has no eligible player")
        best = max(scores[i] for i in eligible)
        winners = [i for i in eligible if scores[i] == best]
        winners.sort(key=lambda i: (i - button - 1) % n)
        share, odd = divmod(amount, len(winners))
        for j, i in enumerate(winners):
            awards[i] += share + (j < odd)
        previous = level
    return awards


class Table:
    def __init__(self, agents, seed=7, stacks=None, payouts=(), table_id="simulation", enforce_timeout=False):
        self.agents = agents
        self.n = len(agents)
        self.stacks = list(stacks) if stacks is not None else [2000] * self.n
        self.initial_total = sum(self.stacks)
        self.rng = random.Random(seed)
        self.button = self.n - 1
        self.hand_id = -1
        self.event_id = 0
        self.payouts = tuple(payouts)
        self.table_id = table_id
        self.timings = [[] for _ in agents]
        self.timeouts = [0] * self.n
        self.enforce_timeout = enforce_timeout
        self.observations = []
        self.capture = False

    def next_seat(self, seat, eligible):
        for step in range(1, self.n + 1):
            candidate = (seat + step) % self.n
            if candidate in eligible:
                return candidate
        raise AssertionError("No eligible next seat")

    def play_hand(self, big_blind=20, deck=None):
        live = {i for i, stack in enumerate(self.stacks) if stack > 0}
        if len(live) < 2:
            return None
        self.hand_id += 1
        self.button = self.next_seat(self.button, live)
        sb = self.button if len(live) == 2 else self.next_seat(self.button, live)
        bb = self.next_seat(sb, live)
        if deck is None:
            deck = list(range(52))
            self.rng.shuffle(deck)
        else:
            deck = list(deck)
        holes = [[] for _ in range(self.n)]
        for _ in range(2):
            for seat in sorted(live):
                holes[seat].append(deck.pop())
        folded = [i not in live for i in range(self.n)]
        committed = [0] * self.n
        bets = [0] * self.n
        board = []
        events = []
        decisions = 0

        def pay(seat, amount):
            amount = min(self.stacks[seat], amount)
            if amount < 0:
                raise AssertionError("Negative contribution")
            self.stacks[seat] -= amount
            committed[seat] += amount
            bets[seat] += amount

        pay(sb, max(1, big_blind // 2))
        pay(bb, big_blind)
        for street in ("preflop", "flop", "turn", "river"):
            if street != "preflop":
                board.extend(deck.pop() for _ in range(3 if street == "flop" else 1))
                bets = [0] * self.n
            current = max(bets)
            # A short big blind does not reduce the minimum bring-in.
            if street == "preflop":
                current = max(current, big_blind)
            last_full_raise = big_blind
            pending = {i for i in live if not folded[i] and self.stacks[i] > 0}
            acted_at = {}
            cursor = bb if street == "preflop" else self.button
            iterations = 0
            while pending:
                iterations += 1
                if iterations > 500:
                    raise AssertionError("Betting round did not terminate")
                survivors = [i for i in live if not folded[i]]
                if len(survivors) == 1:
                    break
                pending = {i for i in pending if not folded[i] and self.stacks[i] > 0}
                if not pending:
                    break
                seat = self.next_seat(cursor, pending)
                cursor = seat
                other_can_act = any(self.stacks[i] > 0 for i in survivors if i != seat)
                # With only one player able to act, there is no nominal
                # bring-in to complete above the actual opposing all-ins.
                # In particular, never allow folding an already unmatched SB
                # to a phantom full big blind when the BB posted short.
                required = current if other_can_act else max(bets[i] for i in survivors if i != seat)
                cost = min(self.stacks[seat], max(0, required - bets[seat]))
                if not other_can_act and cost == 0:
                    pending.remove(seat)
                    continue
                legal = ["fold", "call"] if cost else ["check"]
                reopened = seat not in acted_at or current - acted_at[seat] >= last_full_raise
                if other_can_act and self.stacks[seat] > cost and reopened:
                    legal.append("raise")
                cap = committed[seat] + cost
                eligible = sum(min(x, cap) for x in committed)
                obs = Observation(tuple(holes[seat]), tuple(board),
                                  tuple(Player(i, self.stacks[i], bets[i], committed[i], folded[i], self.stacks[i] == 0)
                                        for i in range(self.n)),
                                  seat, self.button, sum(committed), cost,
                                  current + last_full_raise, bets[seat] + self.stacks[seat], tuple(legal),
                                  big_blind, self.hand_id, self.table_id, tuple(events), self.payouts, eligible)
                if self.capture:
                    self.observations.append((seat, obs))
                start = time.perf_counter_ns()
                action = self.agents[seat].act(obs)
                elapsed = time.perf_counter_ns() - start
                self.timings[seat].append(elapsed)
                if elapsed > 1_000_000:
                    self.timeouts[seat] += 1
                    if self.enforce_timeout:
                        action = Action("check" if "check" in legal else "fold")
                if action.action not in legal:
                    raise AssertionError(f"Illegal action {action} for {legal}")
                pending.remove(seat)
                before = current
                if action.action == "fold":
                    folded[seat] = True
                elif action.action == "call":
                    pay(seat, cost)
                elif action.action == "raise":
                    target = action.amount
                    maximum = bets[seat] + self.stacks[seat]
                    if not (current < target <= maximum and (target >= current + last_full_raise or target == maximum)):
                        raise AssertionError(f"Illegal raise {target}; current {current}, min {current+last_full_raise}, max {maximum}")
                    pay(seat, target - bets[seat])
                    current = target
                    if current - before >= last_full_raise:
                        last_full_raise = current - before
                    pending |= {i for i in survivors if i != seat and self.stacks[i] > 0 and bets[i] < current}
                acted_at[seat] = current
                events.append(Event(self.event_id, seat, street, action.action, action.amount if action.action == "raise" else cost, cost))
                self.event_id += 1
                decisions += 1
            survivors = [i for i in live if not folded[i]]
            if len(survivors) == 1:
                self.stacks[survivors[0]] += sum(committed)
                break
        else:
            scores = [evaluate(holes[i] + board) if not folded[i] else -1 for i in range(self.n)]
            awards = settle(committed, folded, scores, self.button)
            self.stacks = [a + b for a, b in zip(self.stacks, awards)]
        if sum(self.stacks) != self.initial_total or min(self.stacks) < 0:
            raise AssertionError("Chip conservation failure")
        return HandResult(self.stacks.copy(), events, board, committed, folded, decisions)

