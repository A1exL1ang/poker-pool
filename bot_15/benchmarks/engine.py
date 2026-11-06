"""Independent local NLHE test engine. Not an official contest simulator.

The default doubling blind schedule is an explicit experimental assumption.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field


def evaluate(cards):
    """Histogram 5--7 card rank; lexicographic tuples, larger is better."""
    counts = [0] * 15
    suits = [[] for _ in range(4)]
    for card in cards:
        r = card // 4 + 2
        counts[r] += 1
        suits[card % 4].append(r)
    ranks = [r for r in range(14, 1, -1) if counts[r]]

    def straight(rs):
        mask = sum(1 << r for r in set(rs))
        if mask & (1 << 14):
            mask |= 2
        for high in range(14, 5 - 1, -1):
            if mask & (31 << (high - 4)) == 31 << (high - 4):
                return high
        return 0

    flush = next((sorted(s, reverse=True) for s in suits if len(s) >= 5), None)
    if flush:
        high = straight(flush)
        if high:
            return (8, high)
    quads = [r for r in ranks if counts[r] == 4]
    if quads:
        return (7, quads[0], next(r for r in ranks if r != quads[0]))
    trips = [r for r in ranks if counts[r] >= 3]
    if trips:
        pairs = [r for r in ranks if r != trips[0] and counts[r] >= 2]
        if pairs:
            return (6, trips[0], pairs[0])
    if flush:
        return (5, *flush[:5])
    high = straight(ranks)
    if high:
        return (4, high)
    if trips:
        return (3, trips[0], *[r for r in ranks if r != trips[0]][:2])
    pairs = [r for r in ranks if counts[r] >= 2]
    if len(pairs) >= 2:
        return (2, *pairs[:2], next(r for r in ranks if r not in pairs[:2]))
    if pairs:
        return (1, pairs[0], *[r for r in ranks if r != pairs[0]][:3])
    return (0, *ranks[:5])


def icm(stacks, payouts):
    """Exact independent chip model expectation, <=8 players, no cash assumptions."""
    n = len(stacks)
    payouts = tuple(payouts) + (0.0,) * max(0, n - len(payouts))
    values = [0.0] * n
    probabilities = {0: 1.0}
    total = sum(stacks)
    for place in range(n):
        next_prob = {}
        for mask, prob in probabilities.items():
            remaining = total - sum(stacks[i] for i in range(n) if mask >> i & 1)
            indices = [i for i in range(n) if not mask >> i & 1]
            for i in indices:
                share = stacks[i] / remaining if remaining else 1 / len(indices)
                p = prob * share
                values[i] += p * payouts[place]
                newmask = mask | (1 << i)
                next_prob[newmask] = next_prob.get(newmask, 0) + p
        probabilities = next_prob
    return values


def next_hand_positions(stacks, previous_bb):
    """Dead-button progression: next surviving seat after prior BB posts BB.

    Empty button positions within the same gap are strategically equivalent.
    A busted previous BB produces a dead SB, unless transitioning to heads-up.
    """
    n = len(stacks)
    alive = [i for i, stack in enumerate(stacks) if stack > 0]
    bb = next((previous_bb + k) % n for k in range(1, n + 1)
              if stacks[(previous_bb + k) % n] > 0)
    if len(alive) == 2:
        button = next(i for i in alive if i != bb)
        return button, button, bb
    sb = previous_bb if stacks[previous_bb] > 0 else None
    return (previous_bb - 1) % n, sb, bb


@dataclass
class Hand:
    stacks: list
    agents: list
    button: int
    small_blind: int
    big_blind: int
    seed: int
    hand_id: int = 0
    payouts: list | None = None
    decision_limit_ns: int | None = None
    observer: object = None
    history: list = field(default_factory=list)
    invalid_actions: int = 0
    errors: int = 0
    overruns: int = 0
    decisions: int = 0
    blind_seats: tuple | None = None

    def __post_init__(self):
        self.n = len(self.stacks)
        self.initial = self.stacks[:]
        self.total = sum(self.stacks)
        self.in_hand = [s > 0 for s in self.stacks]
        self.contributions = [0] * self.n
        self.street_bets = [0] * self.n
        self.board = []
        self.hole = [[] for _ in self.stacks]
        self.street = 0
        self.deck = list(range(52))
        random.Random(self.seed).shuffle(self.deck)
        self.high = self.big_blind
        self.last_full_raise = self.big_blind
        self.last_acted = [None] * self.n

    def next_seat(self, start, predicate=None):
        if predicate is None:
            predicate = lambda i: self.in_hand[i]
        return next(((start + k) % self.n for k in range(1, self.n + 1)
                     if predicate((start + k) % self.n)), None)

    def put(self, seat, amount):
        amount = max(0, min(int(amount), self.stacks[seat]))
        self.stacks[seat] -= amount
        self.street_bets[seat] += amount
        self.contributions[seat] += amount
        return amount

    def min_raise_to(self):
        return self.high + self.last_full_raise

    def can_raise(self, seat):
        if self.stacks[seat] + self.street_bets[seat] <= self.high:
            return False
        if not any(self.in_hand[i] and self.stacks[i] > 0
                   and self.stacks[i] + self.street_bets[i] > self.high
                   for i in range(self.n) if i != seat):
            return False
        acted = self.last_acted[seat]
        return acted is None or self.high - acted >= self.last_full_raise

    def observation(self, seat):
        order = self.preflop_order
        pos = order.index(seat)
        return dict(hero=seat, button=self.button, hand_id=self.hand_id, street=self.street,
                    hole=self.hole[seat][:], board=self.board[:], stacks=self.stacks[:],
                    in_hand=self.in_hand[:], street_bets=self.street_bets[:],
                    contributions=self.contributions[:], pot=sum(self.contributions),
                    big_blind=self.big_blind, small_blind=self.small_blind,
                    to_call=max(0, self.high - self.street_bets[seat]),
                    min_raise_to=self.min_raise_to(),
                    max_raise_to=self.street_bets[seat] + self.stacks[seat],
                    can_raise=self.can_raise(seat),
                    position=sum(self.in_hand[i] for i in order[pos + 1:]),
                    history=[h.copy() for h in self.history],
                    **({'payouts': self.payouts[:]} if self.payouts is not None else {}))

    def apply(self, seat, action):
        call = max(0, self.high - self.street_bets[seat])
        kind = action.get('action') if isinstance(action, dict) else None
        amount = action.get('amount', 0) if isinstance(action, dict) else 0
        valid = (kind == 'fold' or kind == 'check' and call == 0 or kind == 'call' and call > 0)
        if kind == 'raise':
            valid = (type(amount) is int and self.can_raise(seat)
                     and self.high < amount <= self.street_bets[seat] + self.stacks[seat]
                     and (amount >= self.min_raise_to()
                          or amount == self.street_bets[seat] + self.stacks[seat]))
        if not valid:
            self.invalid_actions += 1
            kind, amount = ('check', 0) if call == 0 else ('fold', 0)
        pot = sum(self.contributions)
        oldhigh = self.high
        if kind == 'fold':
            self.in_hand[seat] = False
            amount = 0
        elif kind == 'check':
            amount = 0
        elif kind == 'call':
            amount = self.put(seat, call)
        else:
            self.put(seat, amount - self.street_bets[seat])
            self.high = amount
            if amount - oldhigh >= self.last_full_raise:
                self.last_full_raise = amount - oldhigh
        self.last_acted[seat] = self.high
        self.history.append(dict(seat=seat, street=self.street, action=kind,
                                 amount=amount, pot_before=pot, to_call=call))
        assert sum(self.stacks) + sum(self.contributions) == self.total

    def betting(self, first):
        seat = first
        actions = 0
        while sum(self.in_hand) > 1:
            eligible = [i for i in range(self.n) if self.in_hand[i] and self.stacks[i] > 0]
            needs = [i for i in eligible if self.last_acted[i] is None or self.street_bets[i] < self.high]
            if not needs:
                return
            if len(eligible) == 1 and self.street_bets[eligible[0]] >= self.high:
                return
            if seat not in needs:
                seat = self.next_seat(seat, lambda i: i in needs)
            obs = self.observation(seat)
            start = time.perf_counter_ns()
            try:
                action = self.agents[seat].act(obs)
            except Exception:
                self.errors += 1
                action = {'action': 'check' if obs['to_call'] == 0 else 'fold', 'amount': 0}
            elapsed = time.perf_counter_ns() - start
            self.decisions += 1
            if self.decision_limit_ns is not None and elapsed > self.decision_limit_ns:
                self.overruns += 1
                action = {'action': 'check' if obs['to_call'] == 0 else 'fold', 'amount': 0}
            if self.observer:
                self.observer(obs, action, elapsed)
            self.apply(seat, action)
            seat = (seat + 1) % self.n
            actions += 1
            if actions > 10000:
                raise RuntimeError('Betting round failed to terminate')

    def settle(self):
        live = [i for i in range(self.n) if self.in_hand[i]]
        ranks = {i: evaluate(self.hole[i] + self.board) for i in live} if len(live) > 1 else {}
        previous = 0
        self.pots = []
        for level in sorted(set(self.contributions) - {0}):
            contributors = [i for i, value in enumerate(self.contributions) if value >= level]
            size = (level - previous) * len(contributors)
            eligible = [i for i in contributors if self.in_hand[i]]
            if len(contributors) == 1:
                winners = contributors  # Uncalled excess, including artificial test fixtures.
            elif len(live) == 1:
                winners = live
            else:
                assert eligible, 'Pot has no eligible player'
                best = max(ranks[i] for i in eligible)
                winners = [i for i in eligible if ranks[i] == best]
            # Odd chips go clockwise from left of the button among tied winners.
            winners.sort(key=lambda i: (i - self.button - 1) % self.n)
            share, odd = divmod(size, len(winners))
            for k, i in enumerate(winners):
                self.stacks[i] += share + (k < odd)
            self.pots.append(dict(amount=size, eligible=eligible, winners=winners[:]))
            previous = level
        assert sum(self.stacks) == self.total
        return self.stacks

    def play(self):
        alive = [i for i, s in enumerate(self.stacks) if s > 0]
        if len(alive) < 2:
            return self.stacks
        if self.button not in alive and self.blind_seats is None:
            self.button = self.next_seat(self.button)
        if self.blind_seats is None:
            sb = self.button if len(alive) == 2 else self.next_seat(self.button)
            bb = self.next_seat(sb)
        else:
            sb, bb = self.blind_seats
        self.sb_seat, self.bb_seat = sb, bb
        first = sb if len(alive) == 2 else self.next_seat(bb)
        self.preflop_order = []
        i = first
        while i not in self.preflop_order:
            self.preflop_order.append(i)
            i = self.next_seat(i)
        for _ in range(2):
            for i in alive:
                self.hole[i].append(self.deck.pop())
        if sb is not None:
            self.put(sb, self.small_blind)
        self.put(bb, self.big_blind)
        self.betting(first)
        for street, count in ((1, 3), (2, 1), (3, 1)):
            if sum(self.in_hand) == 1:
                break
            self.deck.pop()  # burn
            self.board.extend(self.deck.pop() for _ in range(count))
            self.street = street
            self.street_bets = [0] * self.n
            self.high = 0
            self.last_full_raise = self.big_blind
            self.last_acted = [None] * self.n
            self.betting(self.next_seat(self.button))
        return self.settle()


def play_table(agents, seed, starting_stack=2000, max_hands=200, blind_schedule=None,
               payouts=None, decision_limit_ns=None, observer=None):
    """Reset every agent per table. Default schedule is LOCAL ASSUMPTION only."""
    n = len(agents)
    if not 2 <= n <= 8:
        raise ValueError('Local simulator supports 2--8 players')
    for agent in agents:
        if hasattr(agent, 'reset'):
            agent.reset()
    if blind_schedule is None:
        blind_schedule = [(10 * 2 ** k, 20 * 2 ** k) for k in range(8)]
    if payouts is not None and len(payouts) != n:
        raise ValueError('Supply one payout per initial seat')
    stacks = [starting_stack] * n
    button = seed % n
    blind_seats = None
    points = [0.0] * n
    finish = [None] * n
    stats = dict(decisions=0, errors=0, invalid_actions=0, overruns=0)
    played = 0
    for hand_id in range(max_hands):
        live = [i for i, s in enumerate(stacks) if s > 0]
        if len(live) <= 1:
            break
        initial = stacks[:]
        sb, bb = blind_schedule[min(hand_id // 25, len(blind_schedule) - 1)]
        hand = Hand(stacks, agents, button, sb, bb,
                    (seed * 0x9E3779B97F4A7C15 + hand_id * 0xD1B54A32D192ED03) & ((1 << 64) - 1),
                    hand_id, payouts, decision_limit_ns, observer, blind_seats=blind_seats)
        hand.play()
        played += 1
        for key in stats:
            stats[key] += getattr(hand, key)
        busted = [i for i in live if stacks[i] == 0]
        # Simultaneous eliminations rank by stack at start of hand; ties split ranks/points.
        remaining_rank = len(live)
        for initial_stack in sorted({initial[i] for i in busted}):
            tied = [i for i in busted if initial[i] == initial_stack]
            rank_indices = list(range(remaining_rank - len(tied), remaining_rank))
            for i in tied:
                finish[i] = sum(k + 1 for k in rank_indices) / len(tied)
                if payouts is not None:
                    points[i] = sum(payouts[k] for k in rank_indices) / len(tied)
            remaining_rank -= len(tied)
        if sum(s > 0 for s in stacks) >= 2:
            button, next_sb, next_bb = next_hand_positions(stacks, hand.bb_seat)
            blind_seats = (next_sb, next_bb)
    live = [i for i, s in enumerate(stacks) if s > 0]
    if payouts is not None:
        shares = icm([stacks[i] for i in live], payouts[:len(live)])
        for i, share in zip(live, shares):
            points[i] = share
        assert abs(sum(points) - sum(payouts)) < 1e-7
    for i in live:
        finish[i] = 1 if len(live) == 1 else None
    return dict(stacks=stacks, points=points if payouts is not None else None,
                finish=finish, hands=played, survivors=len(live), **stats)
