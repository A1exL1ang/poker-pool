"""Independent local NLHE simulator, not the contest's official simulator.

All money is integer simulation chips. See CONTRACT.md for observations/actions.
We follow TDA-style raise reopening and dead-button progression. Configurable
blind levels and payout vectors are experimental assumptions, never official.

Primary rules references consulted during implementation:
https://www.pokertda.com/view-poker-tda-rules/ (2026 rules, 21/34/36/45/49)
https://www.pokertda.com/forum/index.php?topic=1456.0 (short opening all-ins)
In particular, a postflop all-in bet of 5 with a minimum bet of 20 permits a
full raise to 25, not a completion to 20. A short preflop big blind instead
retains the full nominal opening price while multiple players can contest it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Callable, Sequence


def evaluate(cards: Sequence[int]) -> tuple[int, ...]:
    """Exact best-five value for 5..7 distinct integer cards; larger is better."""
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError("evaluate requires 5..7 distinct cards")
    counts = [0] * 13
    suits = [[], [], [], []]
    for card in cards:
        if not isinstance(card, int) or not 0 <= card < 52:
            raise ValueError("card outside 0..51")
        rank, suit = divmod(card, 4)
        counts[rank] += 1
        suits[suit].append(rank)
    ranks = [r for r in range(12, -1, -1) if counts[r]]

    def straight(rs):
        present = set(rs)
        for high in range(12, 3, -1):
            if all(high - d in present for d in range(5)):
                return high
        return 3 if {12, 0, 1, 2, 3} <= present else -1

    flush = next((sorted(s, reverse=True) for s in suits if len(s) >= 5), None)
    if flush is not None:
        high = straight(flush)
        if high >= 0:
            return (8, high)
    quads = [r for r in ranks if counts[r] == 4]
    if quads:
        return (7, quads[0], next(r for r in ranks if r != quads[0]))
    trips = [r for r in ranks if counts[r] >= 3]
    if trips:
        pairs = [r for r in ranks if r != trips[0] and counts[r] >= 2]
        if pairs:
            return (6, trips[0], pairs[0])
    if flush is not None:
        return (5, *flush[:5])
    high = straight(ranks)
    if high >= 0:
        return (4, high)
    if trips:
        return (3, trips[0], *[r for r in ranks if r != trips[0]][:2])
    pairs = [r for r in ranks if counts[r] >= 2]
    if len(pairs) >= 2:
        return (2, *pairs[:2], next(r for r in ranks if r not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *[r for r in ranks if r != pairs[0]][:3])
    return (0, *ranks[:5])


def icm(stacks: Sequence[int | float], payouts: Sequence[int | float]) -> list[float]:
    """Exact independent chip model by probability propagation over subsets."""
    n = len(stacks)
    if len(payouts) < n or any(x < 0 for x in stacks):
        raise ValueError("invalid ICM inputs")
    if any(not math.isfinite(x) for x in (*stacks, *payouts)):
        raise ValueError("nonfinite ICM input")
    if any(payouts[i] < payouts[i + 1] for i in range(len(payouts) - 1)):
        raise ValueError("payouts must be nonincreasing")
    positive = [i for i, x in enumerate(stacks) if x > 0]
    zero = [i for i, x in enumerate(stacks) if x == 0]
    answer = [0.0] * n
    m = len(positive)
    if zero:
        residual = sum(payouts[m:n]) / len(zero)
        for i in zero:
            answer[i] = residual
    if not m:
        return answer
    weights = [stacks[i] for i in positive]
    size = 1 << m
    sums = [0.0] * size
    for mask in range(1, size):
        bit = mask & -mask
        sums[mask] = sums[mask ^ bit] + weights[bit.bit_length() - 1]
    probs = [0.0] * size
    probs[-1] = 1.0
    for mask in range(size - 1, 0, -1):
        rank = m - mask.bit_count()
        available = mask
        while available:
            bit = available & -available
            available ^= bit
            j = bit.bit_length() - 1
            p = probs[mask] * weights[j] / sums[mask]
            answer[positive[j]] += p * payouts[rank]
            probs[mask ^ bit] += p
    return answer


def award_pots(committed, folded, ranks, button):
    """Return chip awards and side-pot detail, including unmatched refunds.

    An uncontested highest contribution is a refund regardless of hand rank.
    Each tied side pot assigns its odd chips clockwise left of the button.
    """
    n = len(committed)
    awards = [0] * n
    pots = []
    previous = 0
    for level in sorted(set(c for c in committed if c > 0)):
        contributors = [i for i, c in enumerate(committed) if c >= level]
        amount = (level - previous) * len(contributors)
        previous = level
        if len(contributors) == 1:
            winner = contributors[0]
            awards[winner] += amount
            pots.append({"amount": amount, "eligible": [winner], "winners": [winner], "refund": True})
            continue
        eligible = [i for i in contributors if not folded[i]]
        if not eligible:
            raise ValueError("side pot has no eligible live player")
        best = max(ranks[i] for i in eligible)
        winners = [i for i in eligible if ranks[i] == best]
        winners.sort(key=lambda i: (i - button - 1) % n)
        share, odd = divmod(amount, len(winners))
        for j, i in enumerate(winners):
            awards[i] += share + (j < odd)
        pots.append({"amount": amount, "eligible": eligible, "winners": winners, "refund": False})
    return awards, pots


@dataclass
class Player:
    seat: int
    stack: int
    bet: int = 0
    committed: int = 0
    folded: bool = False
    all_in: bool = False


class Hand:
    """A step-driven hand, useful both for simulation and rules unit tests."""

    def __init__(self, stacks, button=0, small_blind=10, big_blind=20, *,
                 rng=None, deck=None, hand_id=0, payouts=None, hands_remaining=None,
                 table_id=None, sb_seat=None, bb_seat=None):
        if not 2 <= sum(s > 0 for s in stacks) <= 8:
            raise ValueError("a hand needs two through eight funded players")
        if any(not isinstance(s, int) or s < 0 for s in stacks):
            raise ValueError("stacks must be nonnegative integer chips")
        if not 0 < small_blind <= big_blind:
            raise ValueError("invalid blinds")
        self.n = len(stacks)
        self.initial_total = sum(stacks)
        self.initial_stacks = list(stacks)
        self.players = [Player(i, s, folded=(s == 0)) for i, s in enumerate(stacks)]
        self.button = button % self.n
        self.sb, self.bb = small_blind, big_blind
        self.hand_id = hand_id
        self.payouts = list(payouts) if payouts is not None else None
        self.hands_remaining = hands_remaining
        self.table_id = table_id
        self.rng = rng or random.Random()
        self.deck = list(range(52)) if deck is None else list(deck)
        if len(self.deck) != 52 or set(self.deck) != set(range(52)):
            raise ValueError("deck must be a permutation of all 52 cards")
        if deck is None:
            self.rng.shuffle(self.deck)
        self.deck_index = 0
        self.holes = [[] for _ in stacks]
        self.board = []
        self.burns = []
        self.history = []
        self.street = 0
        self.done = False
        self.pots = []
        self.awards = [0] * self.n
        self.refunds = [0] * self.n
        alive = [p.seat for p in self.players if not p.folded]
        if len(alive) == 2:
            if self.button not in alive:
                self.button = self._next(self.button, alive)
            self.sb_seat = self.button
            self.bb_seat = self._next(self.button, alive)
        else:
            # A dead small blind is supplied as an unfunded seat by Table.
            self.sb_seat = self._next(self.button, alive) if sb_seat is None else sb_seat
            self.bb_seat = self._next(self.sb_seat, alive) if bb_seat is None else bb_seat
        dealing = sorted(alive, key=lambda i: (i - self.button - 1) % self.n)
        for _ in range(2):
            for seat in dealing:
                self.holes[seat].append(self._draw())
        if self.players[self.sb_seat].stack:
            self._post(self.sb_seat, small_blind)
        self._post(self.bb_seat, big_blind)
        self.current_bet = big_blind  # A short BB does not reduce the opening price.
        self.last_full_raise = big_blind
        self.last_action_bet = {}
        self.pending = set(self._active())
        self.cursor = (self.bb_seat + 1) % self.n
        self.actor = None
        self._advance()

    def _next(self, seat, candidates):
        return min(candidates, key=lambda i: (i - seat - 1) % self.n)

    def _draw(self):
        c = self.deck[self.deck_index]
        self.deck_index += 1
        return c

    def _post(self, seat, amount):
        p = self.players[seat]
        paid = min(amount, p.stack)
        p.stack -= paid
        p.bet += paid
        p.committed += paid
        p.all_in = p.stack == 0
        self.history.append({"seat": seat, "street": 0, "action": "blind", "amount": p.bet, "paid": paid})

    def _live(self):
        return [p.seat for p in self.players if not p.folded]

    def _active(self):
        return [p.seat for p in self.players if not p.folded and p.stack > 0]

    def _call_target(self):
        if len(self._active()) <= 1:
            return max(p.bet for p in self.players)
        return self.current_bet

    def _advance(self):
        while not self.done:
            live = self._live()
            if len(live) == 1:
                self._finish_uncontested(live[0])
                return
            active = self._active()
            self.pending.intersection_update(active)
            if len(active) <= 1:
                target = max(p.bet for p in self.players)
                self.pending = {i for i in active if self.players[i].bet < target}
            if self.pending:
                self.actor = min(self.pending, key=lambda i: (i - self.cursor) % self.n)
                return
            self._refund_uncalled()
            if self.street == 3:
                self._showdown()
                return
            if len(self._active()) <= 1:
                while self.street < 3:
                    self._deal_street()
                self._showdown()
                return
            self._deal_street()
            for p in self.players:
                p.bet = 0
            self.current_bet = 0
            self.last_full_raise = self.bb
            self.last_action_bet = {}
            self.pending = set(self._active())
            self.cursor = (self.button + 1) % self.n

    def _deal_street(self):
        self.street += 1
        self.burns.append(self._draw())
        for _ in range(3 if self.street == 1 else 1):
            self.board.append(self._draw())

    def _refund_uncalled(self):
        ordered = sorted((p.bet, p.seat) for p in self.players)
        amount = ordered[-1][0] - ordered[-2][0]
        if amount:
            p = self.players[ordered[-1][1]]
            p.bet -= amount
            p.committed -= amount
            p.stack += amount
            p.all_in = False
            self.refunds[p.seat] += amount

    def _finish_uncontested(self, winner):
        amount = sum(p.committed for p in self.players)
        self.players[winner].stack += amount
        self.awards[winner] += amount
        self.pots = [{"amount": amount, "eligible": [winner], "winners": [winner], "refund": False}]
        self.done = True
        self.actor = None
        self.assert_conservation()

    def _showdown(self):
        ranks = {i: evaluate(self.holes[i] + self.board) for i in self._live()}
        self.awards, self.pots = award_pots([p.committed for p in self.players],
                                          [p.folded for p in self.players], ranks, self.button)
        for p, award in zip(self.players, self.awards):
            p.stack += award
        self.done = True
        self.actor = None
        self.assert_conservation()

    def observation(self):
        if self.done or self.actor is None:
            raise ValueError("hand has no pending decision")
        seat = self.actor
        p = self.players[seat]
        target = self._call_target()
        call = min(p.stack, max(0, target - p.bet))
        reopen = (seat not in self.last_action_bet or
                  self.current_bet - self.last_action_bet[seat] >= self.last_full_raise)
        can_raise = (reopen and p.bet + p.stack > self.current_bet and
                     any(i != seat for i in self._active()))
        legal = ["fold", "call"] if call else ["check"]
        if can_raise:
            legal.append("raise")
        obs = {"hand_id": self.hand_id, "hero": seat, "button": self.button,
               "street": self.street, "hole": list(self.holes[seat]), "board": list(self.board),
               "players": [dict(vars(q)) for q in self.players],
               "pot": sum(q.committed for q in self.players),
               "small_blind": self.sb, "big_blind": self.bb, "to_call": call,
               "min_raise_to": self.current_bet + self.last_full_raise,
               "max_raise_to": p.bet + p.stack, "can_raise": can_raise,
               "legal_actions": legal, "history": [dict(a) for a in self.history]}
        if self.payouts is not None:
            obs["payouts"] = list(self.payouts)
        if self.hands_remaining is not None:
            obs["hands_remaining"] = self.hands_remaining
        if self.table_id is not None:
            obs["table_id"] = self.table_id
        return obs

    def valid_action(self, action):
        if self.done or not isinstance(action, dict):
            return False
        obs = self.observation()
        kind = action.get("action")
        if kind not in obs["legal_actions"]:
            return False
        if kind == "raise":
            amount = action.get("amount")
            if type(amount) is not int:
                return False
            return (amount > self.current_bet and amount <= obs["max_raise_to"] and
                    (amount >= obs["min_raise_to"] or amount == obs["max_raise_to"]))
        return True

    def step(self, action):
        """Apply a strictly validated action; illegal actions raise ValueError."""
        if not self.valid_action(action):
            raise ValueError(f"illegal action {action!r}")
        seat = self.actor
        p = self.players[seat]
        kind = action["action"]
        old_bet = self.current_bet
        paid = 0
        amount = 0
        self.pending.remove(seat)
        if kind == "fold":
            p.folded = True
        elif kind in ("call", "raise"):
            target = action["amount"] if kind == "raise" else self._call_target()
            paid = min(p.stack, target - p.bet)
            p.stack -= paid
            p.bet += paid
            p.committed += paid
            p.all_in = p.stack == 0
            amount = p.bet
            if kind == "raise":
                self.current_bet = p.bet
                increment = p.bet - old_bet
                if increment >= self.last_full_raise:
                    self.last_full_raise = increment
                self.pending.update(i for i in self._active() if i != seat)
        self.last_action_bet[seat] = self.current_bet
        self.history.append({"seat": seat, "street": self.street, "action": kind,
                             "amount": amount, "paid": paid})
        self.cursor = (seat + 1) % self.n
        self._advance()
        self.assert_conservation()

    def assert_conservation(self):
        assert all(p.stack >= 0 and p.bet >= 0 and p.committed >= p.bet for p in self.players)
        total = sum(p.stack for p in self.players)
        if not self.done:
            total += sum(p.committed for p in self.players)
        assert total == self.initial_total, (total, self.initial_total)


# Deliberately illustrative, configurable, and not an official blind schedule.
DEFAULT_LEVELS = ((10, 20), (15, 30), (25, 50), (40, 80),
                  (60, 120), (100, 200), (150, 300), (250, 500))


class Table:
    """Run a bounded local sit-and-go with anonymous per-table agent state."""

    def __init__(self, agents, seed=0, *, stacks=None, levels=DEFAULT_LEVELS,
                 hands_per_level=25, max_hands=200, payouts=None, time_limit_ns=None,
                 collect_observations=False):
        if not 2 <= len(agents) <= 8:
            raise ValueError("table requires 2..8 agents")
        self.agents = agents
        self.seed = seed
        self.rng = random.Random(seed)
        self.stacks = [2000] * len(agents) if stacks is None else list(stacks)
        if len(self.stacks) != len(agents) or any(s <= 0 for s in self.stacks):
            raise ValueError("all table seats must start funded")
        self.total = sum(self.stacks)
        self.levels = tuple(levels)
        self.hands_per_level = hands_per_level
        self.max_hands = max_hands
        self.payouts = list(payouts) if payouts is not None else None
        if self.payouts is not None:
            icm(self.stacks, self.payouts)  # Validate payout shape/order.
        self.time_limit_ns = time_limit_ns
        self.collect_observations = collect_observations
        self.observations = []
        self.latencies = [[] for _ in agents]
        self.errors = [0] * len(agents)
        self.invalid_actions = [0] * len(agents)
        self.overruns = [0] * len(agents)
        self.observed_overruns = [0] * len(agents)
        self.scores = [0.0] * len(agents)
        self.placements = [None] * len(agents)
        self.hand_results = []
        for agent in agents:
            if hasattr(agent, "reset"):
                agent.reset()

    def run(self):
        n = len(self.agents)
        button = self.rng.randrange(n)
        previous_sb = None
        previous_bb = None
        hands_played = 0
        for hand_id in range(self.max_hands):
            alive = [i for i, s in enumerate(self.stacks) if s > 0]
            if len(alive) < 2:
                break
            if previous_bb is None:
                sb = bb = None
            else:
                bb = min(alive, key=lambda i: (i - previous_bb - 1) % n)
                if len(alive) == 2:
                    button = next(i for i in alive if i != bb)
                    sb = button
                else:
                    button = previous_sb
                    sb = previous_bb
            level = self.levels[min(hand_id // self.hands_per_level, len(self.levels) - 1)]
            starting = list(self.stacks)
            hand = Hand(self.stacks, button, *level, rng=self.rng, hand_id=hand_id,
                        payouts=self.payouts, hands_remaining=self.max_hands - hand_id,
                        table_id=self.seed, sb_seat=sb, bb_seat=bb)
            previous_sb, previous_bb = hand.sb_seat, hand.bb_seat
            button = hand.button
            steps = 0
            while not hand.done:
                steps += 1
                if steps > 10000:
                    raise RuntimeError("unexpected nonterminating hand")
                obs = hand.observation()
                seat = obs["hero"]
                # Use a fresh copy for the agent, so observation mutation cannot
                # affect legality checks or the saved benchmark corpus.
                if self.collect_observations:
                    self.observations.append(hand.observation())
                fallback = {"action": "check" if "check" in obs["legal_actions"] else "fold"}
                start = time.perf_counter_ns()
                try:
                    action = self.agents[seat].act(obs)
                except Exception:
                    self.errors[seat] += 1
                    action = fallback
                elapsed = time.perf_counter_ns() - start
                self.latencies[seat].append(elapsed)
                if elapsed > 1_000_000:
                    self.observed_overruns[seat] += 1
                if self.time_limit_ns is not None and elapsed > self.time_limit_ns:
                    self.overruns[seat] += 1
                    action = fallback
                if not hand.valid_action(action):
                    self.invalid_actions[seat] += 1
                    action = fallback
                hand.step(action)
            self.stacks = [p.stack for p in hand.players]
            assert sum(self.stacks) == self.total
            eliminated = [i for i in alive if self.stacks[i] == 0]
            remaining = sum(s > 0 for s in self.stacks)
            # Equal starting stacks busting together split the occupied place
            # points. This tie rule is an explicit local assumption.
            eliminated.sort(key=lambda i: (-starting[i], i))
            offset = 0
            while offset < len(eliminated):
                end = offset + 1
                while end < len(eliminated) and starting[eliminated[end]] == starting[eliminated[offset]]:
                    end += 1
                places = list(range(remaining + offset + 1, remaining + end + 1))
                average_place = sum(places) / len(places)
                points = (sum(self.payouts[p - 1] for p in places) / len(places)
                          if self.payouts is not None else 0.0)
                for i in eliminated[offset:end]:
                    self.placements[i] = average_place
                    self.scores[i] = points
                offset = end
            self.hand_results.append({"hand_id": hand_id, "button": hand.button,
                                      "small_blind_seat": hand.sb_seat, "big_blind_seat": hand.bb_seat,
                                      "stacks": list(self.stacks), "eliminated": eliminated,
                                      "actions": sum(a["action"] != "blind" for a in hand.history)})
            hands_played += 1
        survivors = [i for i, s in enumerate(self.stacks) if s > 0]
        if self.payouts is None:
            # Chip share is a neutral auxiliary metric; not contest points.
            self.scores = [s / self.total for s in self.stacks]
        else:
            residual = icm([self.stacks[i] for i in survivors], self.payouts[:len(survivors)])
            for i, score in zip(survivors, residual):
                self.scores[i] = score
        if len(survivors) == 1:
            self.placements[survivors[0]] = 1.0
        return {"seed": self.seed, "hands": hands_played, "stacks": list(self.stacks),
                "scores": list(self.scores), "placements": list(self.placements),
                "survivors": survivors, "errors": list(self.errors),
                "invalid_actions": list(self.invalid_actions), "overruns": list(self.overruns),
                "observed_over_1ms": list(self.observed_overruns),
                "internal_fallbacks": [getattr(a, "fallbacks", 0) for a in self.agents],
                "decisions": [len(t) for t in self.latencies]}
