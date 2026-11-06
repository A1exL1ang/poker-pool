"""Deterministic local no-limit Texas Hold'em simulator.

Strict action validation is the default. Tournament play converts bot errors,
illegal actions, and optionally measured timeouts to check/fold, counting each.
This simulator is independently written and does not claim official parity.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from time import perf_counter_ns

from .schema import Action, Observation, Player
from .icm import icm
try:
    from equity.evaluator import evaluate
except ImportError:
    from .reference import evaluate

STREETS = ("preflop", "flop", "turn", "river")
SYNTHETIC_BLINDS = ((10, 20), (15, 30), (25, 50), (50, 100),
                    (75, 150), (100, 200), (150, 300), (200, 400))
SYNTHETIC_PAYOUTS = (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class IllegalAction(ValueError):
    pass


def next_seat(seat, seats, n):
    """Next member of seats clockwise, including seat only after one lap."""
    for offset in range(1, n + 1):
        candidate = (seat + offset) % n
        if candidate in seats:
            return candidate
    raise ValueError("no occupied seat")


def legal_action(observation, action):
    """Validate exact local action semantics, without silently clamping bets."""
    if isinstance(action, Action):
        action = action.as_dict()
    if not isinstance(action, dict):
        raise IllegalAction("action must be a dict or Action")
    kind = action.get("action")
    if kind not in ("fold", "check", "call", "raise"):
        raise IllegalAction("unknown action")
    if kind == "check" and observation["to_call"]:
        raise IllegalAction("cannot check while facing a bet")
    if kind == "call" and not observation["to_call"]:
        # Common, harmless no-bet call is normalized to check.
        kind = "check"
    amount = action.get("amount", 0)
    if kind == "raise":
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise IllegalAction("raise-to amount must be integer chips")
        if not observation["can_raise"]:
            raise IllegalAction("raising is unavailable or not reopened")
        player = observation["players"][observation["hero"]]
        current_bet = player["bet"] + observation["to_call"]
        if amount <= current_bet or amount > observation["max_raise_to"]:
            raise IllegalAction("raise must increase wager within stack")
        if amount < observation["min_raise_to"] and amount != observation["max_raise_to"]:
            raise IllegalAction("under-minimum raise only permitted all-in")
    return {"action": kind, "amount": amount if kind == "raise" else 0}


class Hand:
    def __init__(self, stacks, button, small_blind=10, big_blind=20, *,
                 hand_id=0, rng=None, deck=None, sb_seat=-1, bb_seat=-1,
                 payouts=None, hands_remaining=None):
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in stacks):
            raise ValueError("nonnegative integer stacks required")
        if len(stacks) < 2 or len(stacks) > 10:
            raise ValueError("two to ten seats required")
        if small_blind < 0 or big_blind <= 0 or small_blind > big_blind:
            raise ValueError("invalid blinds")
        self.n = len(stacks)
        self.stacks = list(stacks)
        self.initial_stacks = list(stacks)
        self.total_chips = sum(stacks)
        self.button = button % self.n
        self.small_blind, self.big_blind = small_blind, big_blind
        self.hand_id, self.payouts, self.hands_remaining = hand_id, payouts, hands_remaining
        self.seated = {i for i, s in enumerate(stacks) if s > 0}
        if len(self.seated) < 2:
            raise ValueError("at least two positive stacks required")
        self.folded = [i not in self.seated for i in range(self.n)]
        self.bets, self.invested = [0] * self.n, [0] * self.n
        self.holes = [[] for _ in stacks]
        self.board, self.history = [], []
        self.street_index = 0
        self.current_bet = big_blind
        self.last_full_raise = big_blind
        self.acted_at = [None] * self.n
        self.done, self.actor = False, None
        self.awards, self.refunds = [0] * self.n, [0] * self.n
        self.pots = []
        self.rng = rng if rng is not None else random.Random()
        if deck is None:
            deck = list(range(52))
            self.rng.shuffle(deck)
        if len(deck) != 52 or set(deck) != set(range(52)):
            raise ValueError("deck must be a permutation of 0..51")
        self.deck, self.deck_position = list(deck), 0
        if len(self.seated) == 2:
            if self.button not in self.seated:
                raise ValueError("heads-up button must be occupied")
            self.sb_seat = self.button if sb_seat == -1 else sb_seat
            self.bb_seat = next_seat(self.button, self.seated, self.n) if bb_seat == -1 else bb_seat
        else:
            self.sb_seat = next_seat(self.button, self.seated, self.n) if sb_seat == -1 else sb_seat
            self.bb_seat = next_seat(self.sb_seat if self.sb_seat is not None else self.button,
                                     self.seated, self.n) if bb_seat == -1 else bb_seat
        if self.bb_seat not in self.seated or (self.sb_seat is not None and self.sb_seat not in self.seated):
            raise ValueError("blind seats must be occupied")
        # Deal clockwise, button last heads-up, exactly as in a physical deal.
        order = sorted(self.seated, key=lambda s: (s - self.button - 1) % self.n)
        for _ in range(2):
            for seat in order:
                self.holes[seat].append(self._draw())
        if self.sb_seat is not None:
            self._blind(self.sb_seat, small_blind, "small_blind")
        self._blind(self.bb_seat, big_blind, "big_blind")
        self.pending = {i for i in self.seated if self.stacks[i] > 0}
        self._advance(self.bb_seat)

    @property
    def street(self):
        return STREETS[self.street_index]

    @property
    def pot(self):
        return sum(self.invested)

    def _draw(self):
        card = self.deck[self.deck_position]
        self.deck_position += 1
        return card

    def _pay(self, seat, amount):
        if not 0 <= amount <= self.stacks[seat]:
            raise AssertionError("invalid internal payment")
        self.stacks[seat] -= amount
        self.bets[seat] += amount
        self.invested[seat] += amount

    def _blind(self, seat, amount, kind):
        payment = min(amount, self.stacks[seat])
        self._pay(seat, payment)
        self.history.append({"seat": seat, "street": self.street, "kind": kind,
                             "amount": payment, "raise_to": self.bets[seat]})

    def _live(self):
        return {i for i in self.seated if not self.folded[i]}

    def _active(self):
        return {i for i in self.seated if not self.folded[i] and self.stacks[i] > 0}

    def _facing(self, seat):
        # If nobody can respond, match only the actual all-in amount. This
        # handles a short big blind without forcing an unnecessary overcall.
        target = self.current_bet
        if len(self._active()) == 1:
            target = max((self.bets[i] for i in self._live() if i != seat), default=0)
        return min(self.stacks[seat], max(0, target - self.bets[seat]))

    def _can_raise(self, seat):
        if self.stacks[seat] <= self._facing(seat):
            return False
        if not any(i != seat for i in self._active()):
            return False
        previous = self.acted_at[seat]
        return previous is None or self.current_bet - previous >= self.last_full_raise

    def observation(self):
        if self.done or self.actor is None:
            raise ValueError("no action pending")
        hero = self.actor
        players = tuple(Player(i, self.stacks[i], self.bets[i], self.invested[i],
                               self.folded[i], self.stacks[i] == 0 and not self.folded[i])
                        for i in range(self.n))
        return Observation(self.hand_id, self.street, hero, self.button,
                           self.small_blind, self.big_blind, tuple(self.holes[hero]),
                           tuple(self.board), players, self.pot, self._facing(hero),
                           self.current_bet + self.last_full_raise,
                           self.bets[hero] + self.stacks[hero], self._can_raise(hero),
                           tuple(self.history), None if self.payouts is None else tuple(self.payouts),
                           self.hands_remaining).as_dict()

    def apply(self, action):
        observation = self.observation()
        action = legal_action(observation, action)
        seat, kind = self.actor, action["action"]
        self.pending.discard(seat)
        payment = 0
        if kind == "fold":
            self.folded[seat] = True
        elif kind == "call":
            payment = observation["to_call"]
            self._pay(seat, payment)
        elif kind == "raise":
            target = action["amount"]
            increment = target - self.current_bet
            payment = target - self.bets[seat]
            self._pay(seat, payment)
            if increment >= self.last_full_raise:
                self.last_full_raise = increment
            self.current_bet = target
            self.pending.update(i for i in self._active() if i != seat and self.bets[i] < target)
        self.acted_at[seat] = self.current_bet
        self.history.append({"seat": seat, "street": self.street, "kind": kind,
                             "amount": payment, "raise_to": self.bets[seat]})
        self.assert_invariants()
        self._advance(seat)

    def _advance(self, last_actor):
        while not self.done:
            live, active = self._live(), self._active()
            if len(live) == 1:
                self._settle()
                return
            self.pending.intersection_update(active)
            if len(active) == 1:
                only = next(iter(active))
                if not self._facing(only):
                    self.pending.clear()
            if self.pending:
                self.actor = next_seat(last_actor, self.pending, self.n)
                return
            self._refund_uncalled()
            if self.street_index == 3:
                self._settle()
                return
            self.street_index += 1
            self._draw()  # burn card
            self.board.extend(self._draw() for _ in range(3 if self.street_index == 1 else 1))
            self.bets = [0] * self.n
            self.current_bet = 0
            self.last_full_raise = self.big_blind
            self.acted_at = [None] * self.n
            self.pending = self._active() if len(self._active()) > 1 else set()
            last_actor = self.button

    def _refund_uncalled(self):
        """Return a unique unmatched current-street contribution."""
        order = sorted(range(self.n), key=lambda i: self.bets[i], reverse=True)
        highest, second = self.bets[order[0]], self.bets[order[1]]
        if highest > second:
            seat, excess = order[0], highest - second
            self.stacks[seat] += excess
            self.bets[seat] -= excess
            self.invested[seat] -= excess
            self.refunds[seat] += excess

    def _settle(self):
        self._refund_uncalled()
        live = self._live()
        if len(live) == 1:
            winner = next(iter(live))
            self.awards[winner] = self.pot
            self.pots = [{"amount": self.pot, "eligible": [winner], "winners": [winner]}]
        else:
            if len(self.board) != 5:
                raise AssertionError("showdown needs a five-card board")
            strengths = {i: evaluate(self.holes[i] + self.board) for i in live}
            previous = 0
            for level in sorted(set(self.invested) - {0}):
                contributors = [i for i in self.seated if self.invested[i] >= level]
                amount = (level - previous) * len(contributors)
                eligible = [i for i in contributors if i in live]
                if not eligible:
                    raise AssertionError("side pot has no eligible player")
                best = max(strengths[i] for i in eligible)
                winners = [i for i in eligible if strengths[i] == best]
                # Odd chips go first to the tied winner left of the button.
                winners.sort(key=lambda i: (i - self.button - 1) % self.n)
                base, odd = divmod(amount, len(winners))
                for index, i in enumerate(winners):
                    self.awards[i] += base + (index < odd)
                self.pots.append({"amount": amount, "eligible": eligible, "winners": winners})
                previous = level
        for i, award in enumerate(self.awards):
            self.stacks[i] += award
        self.done, self.actor = True, None
        self.assert_invariants()

    def assert_invariants(self):
        if any(s < 0 for s in self.stacks) or any(b < 0 for b in self.bets):
            raise AssertionError("negative chips")
        if any(b > inv for b, inv in zip(self.bets, self.invested)):
            raise AssertionError("street bet exceeds total contribution")
        accounted = sum(self.stacks) if self.done else sum(self.stacks) + self.pot
        if accounted != self.total_chips:
            raise AssertionError(f"chip conservation violated: {accounted} != {self.total_chips}")


@dataclass
class TournamentResult:
    stacks: list[int]
    points: list[float]
    hands: int
    errors: list[int]
    illegal: list[int]
    timeouts: list[int]
    decisions: list[int]
    latency_ns: list[list[int]]
    eliminations: list[dict]


def tournament(agents, *, seed=0, initial_stack=2000, max_hands=200,
               blind_schedule=SYNTHETIC_BLINDS, hands_per_level=25,
               payouts=None, enforce_timeout=False, timeout_ns=1_000_000,
               capture_latencies=False, observer=None, decision_observer=None,
               expose_payouts=True):
    """Run one independent table; agents should be fresh instances per call.

    Blind levels/payouts are LOCAL SYNTHETIC assumptions, not official values.
    Equal starting-stack simultaneous eliminations share the affected payouts.
    observer(hand) runs after settlement and may collect local diagnostics.
    """
    n = len(agents)
    if n < 2 or n > 10 or max_hands <= 0 or hands_per_level <= 0:
        raise ValueError("invalid tournament dimensions")
    if payouts is None:
        payouts = SYNTHETIC_PAYOUTS if n == 8 else (1.0,) + (0.0,) * (n - 1)
    if len(payouts) != n or any(payouts[i] < payouts[i + 1] for i in range(n - 1)):
        raise ValueError("one nonincreasing payout per seat required")
    if not blind_schedule:
        raise ValueError("at least one blind level required")
    rng = random.Random(seed)
    for agent in agents:
        if hasattr(agent, "reset"):
            agent.reset()
    stacks = [initial_stack] * n
    points = [0.0] * n
    errors, illegal, timeouts, decisions = ([0] * n for _ in range(4))
    latency_ns = [[] for _ in range(n)]
    eliminations = []
    button = rng.randrange(n)
    old_sb_marker = old_bb = None
    hands = 0
    for hand_id in range(max_hands):
        alive = {i for i, s in enumerate(stacks) if s > 0}
        if len(alive) <= 1:
            break
        if hand_id == 0:
            sb_seat = button if n == 2 else next_seat(button, alive, n)
            bb_seat = next_seat(sb_seat, alive, n)
            sb_marker = sb_seat
        elif len(alive) == 2:
            # Move the big blind forward, including the 3 -> 2 transition.
            bb_seat = next_seat(old_bb, alive, n)
            button = next_seat(bb_seat, alive, n)
            sb_seat = sb_marker = button
        else:
            button = old_sb_marker
            sb_marker = old_bb
            sb_seat = sb_marker if sb_marker in alive else None
            bb_seat = next_seat(old_bb, alive, n)
        small, big = blind_schedule[min(hand_id // hands_per_level, len(blind_schedule) - 1)]
        hand = Hand(stacks, button, small, big, hand_id=hand_id, rng=rng,
                    sb_seat=sb_seat, bb_seat=bb_seat, payouts=payouts if expose_payouts else None,
                    hands_remaining=max_hands - hand_id)
        old_sb_marker, old_bb = sb_marker, bb_seat
        while not hand.done:
            seat = hand.actor
            observation = hand.observation()
            decisions[seat] += 1
            fallback = {"action": "fold" if observation["to_call"] else "check"}
            started = perf_counter_ns()
            status = "ok"
            try:
                action = agents[seat].act(observation)
            except Exception:
                errors[seat] += 1
                action = fallback
                status = "error"
            elapsed = perf_counter_ns() - started
            if capture_latencies:
                latency_ns[seat].append(elapsed)
            if elapsed > timeout_ns:
                timeouts[seat] += 1
                if enforce_timeout:
                    action = fallback
                    status = "timeout"
            try:
                hand.apply(action)
            except IllegalAction:
                illegal[seat] += 1
                status = "illegal"
                action = fallback
                hand.apply(fallback)
            if decision_observer is not None:
                decision_observer(observation, action, elapsed, status)
        stacks = hand.stacks[:]
        busted = [i for i in alive if stacks[i] == 0]
        # When multiple players bust together, the larger starting stack
        # finishes higher; exact ties share the occupied finishing payouts.
        survivors = len(alive) - len(busted)
        ordered = sorted(busted, key=lambda i: hand.initial_stacks[i], reverse=True)
        rank = survivors
        while ordered:
            tied = [i for i in ordered if hand.initial_stacks[i] == hand.initial_stacks[ordered[0]]]
            share = sum(payouts[rank:rank + len(tied)]) / len(tied)
            for seat in tied:
                points[seat] = share
                eliminations.append({"seat": seat, "hand": hand_id, "rank_first": rank + 1,
                                     "rank_last": rank + len(tied), "points": share})
            rank += len(tied)
            ordered = [i for i in ordered if i not in tied]
        hands = hand_id + 1
        if observer is not None:
            observer(hand)
    remaining = [i for i, s in enumerate(stacks) if s > 0]
    values = icm([stacks[i] for i in remaining], payouts[:len(remaining)])
    for i, value in zip(remaining, values):
        points[i] = value
    if abs(sum(points) - sum(payouts)) > 1e-8:
        raise AssertionError("payout conservation failed")
    return TournamentResult(stacks, points, hands, errors, illegal, timeouts,
                            decisions, latency_ns, eliminations)
