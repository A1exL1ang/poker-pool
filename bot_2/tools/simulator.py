"""Local validation engine, independently authored; NOT the official simulator.

Standard no-limit betting, short-raise reopening, all-ins, side pots, HU blinds,
and exact ICM allocation at the hand cap. Payouts/blind increments are explicit
experimental parameters. No assumptions about unpublished competition data.
"""

import random
import time
from dataclasses import dataclass
from pokerbot.model import Action, Event, Observation, Player
from pokerbot.cards import evaluate
from pokerbot.icm import equities


@dataclass
class TournamentResult:
    points: list
    stacks: list
    hands: int
    errors: int
    decisions: int
    actions: dict
    timeouts: list


class Table:
    def __init__(self, agents, seed=1, starting_stack=2000, payouts=None,
                 blinds=None, hand_cap=200, strict=True, decision_limit_us=None):
        self.agents = agents
        self.n = len(agents)
        self.rng = random.Random(seed)
        self.stacks = [starting_stack] * self.n
        self.total_chips = sum(self.stacks)
        self.points = [None] * self.n
        self.button = seed % self.n
        self.payouts = tuple(payouts or (self.n - i - 1 for i in range(self.n)))
        # EXAMPLE ONLY. Rules state increments every 25 hands but not amounts.
        self.blinds = tuple(blinds or (20, 30, 50, 80, 120, 200, 320, 500))
        self.hand_cap = hand_cap
        self.strict = strict
        self.errors = self.decisions = 0
        self.actions = {}
        self.timeouts = [0] * self.n
        self.decision_limit_ns = None if decision_limit_us is None else int(decision_limit_us * 1000)
        self.table_id = f"local-{seed}"
        self.completed_hands = 0

    def next_seat(self, seat, seats):
        for k in range(1, self.n + 1):
            s = (seat + k) % self.n
            if s in seats:
                return s
        raise ValueError("no seat")

    def play(self):
        while sum(s > 0 for s in self.stacks) > 1 and self.completed_hands < self.hand_cap:
            self.play_hand()
        alive = [i for i, s in enumerate(self.stacks) if s > 0]
        if len(alive) == 1:
            self.points[alive[0]] = self.payouts[0]
        else:
            values = equities([self.stacks[i] for i in alive], self.payouts[:len(alive)])
            for i, value in zip(alive, values):
                self.points[i] = value
        assert abs(sum(self.points) - sum(self.payouts)) < 1e-7
        return TournamentResult(self.points[:], self.stacks[:], self.completed_hands,
                                self.errors, self.decisions, self.actions.copy(), self.timeouts[:])

    def play_hand(self):
        alive = {i for i, s in enumerate(self.stacks) if s > 0}
        if len(alive) < 2:
            return
        self.starting = self.stacks[:]
        self.button = self.next_seat(self.button, alive)
        self.folded = [i not in alive for i in range(self.n)]
        self.total = [0] * self.n
        self.bets = [0] * self.n
        self.history = []
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.holes = [tuple(deck[2 * i:2 * i + 2]) for i in range(self.n)]
        self.runout = tuple(deck[2 * self.n:2 * self.n + 5])
        self.board = ()
        self.bb = self.blinds[min(self.completed_hands // 25, len(self.blinds) - 1)]
        self.sb_seat = self.button if len(alive) == 2 else self.next_seat(self.button, alive)
        self.bb_seat = self.next_seat(self.sb_seat, alive)
        self.pay(self.sb_seat, max(1, self.bb // 2))
        self.pay(self.bb_seat, self.bb)
        self.round(0, self.next_seat(self.bb_seat, alive), self.bb)
        for street, length in ((1, 3), (2, 4), (3, 5)):
            if sum(not f for f in self.folded) <= 1:
                break
            self.board = self.runout[:length]
            self.bets = [0] * self.n
            live = {i for i in range(self.n) if not self.folded[i]}
            self.round(street, self.next_seat(self.button, live), 0)
        self.settle()
        self.completed_hands += 1
        assert sum(self.stacks) == self.total_chips, (self.stacks, self.total_chips)
        assert all(s >= 0 for s in self.stacks)
        busted = [i for i in alive if self.stacks[i] == 0]
        # Standard tournament tie rule by starting stack; equal starts split
        # the relevant finishing-place points in this local test model.
        remaining = len(alive)
        for amount in sorted({self.starting[i] for i in busted}):
            group = [i for i in busted if self.starting[i] == amount]
            p = sum(self.payouts[remaining - k - 1] for k in range(len(group))) / len(group)
            for i in group:
                self.points[i] = p
            remaining -= len(group)

    def pay(self, seat, amount):
        amount = min(self.stacks[seat], max(0, int(amount)))
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.total[seat] += amount

    def round(self, street, first, initial_current):
        current = initial_current
        last_full = self.bb
        pending = {i for i in range(self.n) if not self.folded[i] and self.stacks[i] > 0}
        reopen = {}
        seat = first
        count = 0
        while pending and sum(not f for f in self.folded) > 1:
            # One player with chips cannot bet against only all-in players.
            if len(pending) == 1:
                last = next(iter(pending))
                opponents_with_chips = any(i != last and not self.folded[i] and self.stacks[i] > 0 for i in range(self.n))
                if not opponents_with_chips:
                    actual = max(self.bets[i] for i in range(self.n) if i != last and not self.folded[i])
                    if self.bets[last] >= actual:
                        break
                    # A nominal full BB is relevant while other players can
                    # contest a side pot. With everyone else all-in, only their
                    # actual highest wager needs matching; excess is inaccessible.
                    current = actual
            if seat not in pending:
                seat = self.next_seat(seat, pending)
            count += 1
            if count > 150:
                raise RuntimeError("betting round failed to terminate")
            call = min(self.stacks[seat], max(0, current - self.bets[seat]))
            can_raise = (seat not in reopen or current >= reopen[seat])
            can_raise &= any(i != seat and not self.folded[i] and self.stacks[i] > 0 for i in range(self.n))
            maximum = self.bets[seat] + self.stacks[seat]
            # A short opening all-in cannot be "completed" to the blind in
            # no-limit: an unacted player must add a full minimum raise.
            minimum = current + last_full
            legal = ("fold", "call") if call else ("check",)
            if can_raise and maximum > current:
                legal += ("raise",)
            obs = Observation(
                hole=self.holes[seat], board=self.board, hero=seat, button=self.button,
                players=tuple(Player(i, self.stacks[i], self.bets[i], self.total[i], self.folded[i], self.starting[i] == 0) for i in range(self.n)),
                pot=sum(self.total), big_blind=self.bb, to_call=call,
                min_raise_to=minimum, max_raise_to=maximum,
                can_raise=bool(can_raise), history=tuple(self.history),
                hand_id=self.completed_hands, table_id=self.table_id,
                payouts=self.payouts, hands_remaining=self.hand_cap - self.completed_hands,
                legal=legal,
            )
            self.decisions += 1
            try:
                started = time.perf_counter_ns() if self.decision_limit_ns is not None else 0
                action = self.agents[seat].act(obs)
                elapsed = time.perf_counter_ns() - started if self.decision_limit_ns is not None else 0
                if self.decision_limit_ns is not None and elapsed > self.decision_limit_ns:
                    self.timeouts[seat] += 1
                    action = Action("fold" if call else "check")
                self.validate_action(obs, action)
            except Exception:
                self.errors += 1
                if self.strict:
                    raise
                action = Action("fold" if call else "check")
            self.actions[action.kind] = self.actions.get(action.kind, 0) + 1
            pending.remove(seat)
            if action.kind == "fold":
                self.folded[seat] = True
            elif action.kind == "call":
                self.pay(seat, call)
            elif action.kind == "raise":
                target = action.amount
                full = target >= minimum
                old = current
                self.pay(seat, target - self.bets[seat])
                current = target
                if full:
                    last_full = current - old
                    pending = {i for i in range(self.n) if i != seat and not self.folded[i] and self.stacks[i] > 0}
                else:
                    pending |= {i for i in range(self.n) if i != seat and not self.folded[i] and self.stacks[i] > 0 and self.bets[i] < current}
            self.history.append(Event(seat, street, action.kind, self.bets[seat] if action.kind in ("raise", "call") else 0))
            reopen[seat] = current + last_full
            if pending:
                seat = self.next_seat(seat, pending)

    @staticmethod
    def validate_action(obs, action):
        assert isinstance(action, Action), action
        assert action.kind in obs.legal, (action, obs)
        if action.kind == "raise":
            hero = next(p for p in obs.players if p.seat == obs.hero)
            assert isinstance(action.amount, int)
            assert action.amount <= obs.max_raise_to
            assert action.amount > hero.street_bet + obs.to_call
            assert action.amount >= obs.min_raise_to or action.amount == obs.max_raise_to
        else:
            assert action.amount == 0

    def settle(self):
        # Return only the unmatched part of the unique largest contribution.
        amounts = sorted(self.total, reverse=True)
        if amounts[0] > amounts[1]:
            seat = self.total.index(amounts[0])
            refund = amounts[0] - amounts[1]
            self.stacks[seat] += refund
            self.total[seat] -= refund
        remaining = [i for i in range(self.n) if not self.folded[i]]
        if len(remaining) == 1:
            self.stacks[remaining[0]] += sum(self.total)
            return
        ranks = {i: evaluate(self.holes[i] + self.runout) for i in remaining}
        previous = 0
        for level in sorted({v for v in self.total if v > 0}):
            tranche = (level - previous) * sum(v >= level for v in self.total)
            eligible = [i for i in remaining if self.total[i] >= level]
            assert eligible, (self.total, self.folded)
            best = max(ranks[i] for i in eligible)
            winners = [i for i in eligible if ranks[i] == best]
            each, odd = divmod(tranche, len(winners))
            for i in winners:
                self.stacks[i] += each
            order = sorted(winners, key=lambda i: (i - self.button - 1) % self.n)
            for i in order[:odd]:
                self.stacks[i] += 1
            previous = level
