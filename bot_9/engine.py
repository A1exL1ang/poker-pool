"""Independent LOCAL no-limit Hold'em simulator, not an official contest API.

Cards are integers ``4 * (rank - 2) + suit``; higher ranks win. Actions are
``{'action': 'fold'|'check'|'call'|'raise', 'amount': total_street_raise_to}``.
The amount field is only used for raises. Pot includes ALL hand contributions,
including the current street. To-call is capped by the hero's remaining stack.
Raises must be at least min_raise_to, except a short all-in to max_raise_to;
can_raise reflects reopening AND whether another player can contest a raise.
Stacks exclude committed chips. Observations expose no opponent private cards.

Tournament payout and blind schedule MUST be supplied: their official values
are unavailable. No rake/antes, one board, no burns (uniform shuffled deck),
moving live button, odd chips clockwise left of button. Simultaneous busts use
start-of-hand stack ordering and split tied places. These are local assumptions.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
import random
import time


def rank_five(cards):
    """Slow independent reference rank. Category then tie breakers, high wins."""
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    groups = sorted(((ranks.count(r), r) for r in set(ranks)), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    unique = sorted(set(ranks), reverse=True)
    straight = unique[0] if len(unique) == 5 and unique[0] - unique[-1] == 4 else 0
    if unique == [14, 5, 4, 3, 2]:
        straight = 5
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if [g[0] for g in groups] == [3, 2]:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for count, r in groups[1:]), reverse=True))
    if groups[0][0] == groups[1][0] == 2:
        return (2, max(groups[0][1], groups[1][1]), min(groups[0][1], groups[1][1]), groups[2][1])
    if groups[0][0] == 2:
        return (1, groups[0][1], *sorted((r for count, r in groups[1:]), reverse=True))
    return (0, *ranks)


def evaluate_reference(cards):
    return max(map(rank_five, combinations(cards, 5)))


def icm(stacks, payouts):
    """Exact independent chip model for positive stacks, rank-order payouts.

    Zero-stack players receive zero here; tournament assigns bust payouts first.
    Remaining payouts are indexed only over surviving players.
    """
    n = len(stacks)
    alive = tuple(i for i, s in enumerate(stacks) if s > 0)
    if len(payouts) < len(alive):
        raise ValueError('Need one payout per live player')

    @lru_cache(None)
    def solve(seats):
        if not seats:
            return (0.0,) * n
        rank = len(alive) - len(seats)
        total = sum(stacks[i] for i in seats)
        out = [0.0] * n
        for winner in seats:
            chance = stacks[winner] / total
            out[winner] += chance * payouts[rank]
            tail = solve(tuple(i for i in seats if i != winner))
            for i in seats:
                out[i] += chance * tail[i]
        return tuple(out)

    return list(solve(alive))


def settle_pots(contributions, folded, ranks, button=0):
    """Return per-seat integer chip awards; handles uncalled chips and odd ties."""
    n = len(contributions)
    awards = [0] * n
    previous = 0
    for level in sorted(set(contributions) - {0}):
        paid = [i for i, c in enumerate(contributions) if c >= level]
        amount = (level - previous) * len(paid)
        previous = level
        if len(paid) == 1:
            awards[paid[0]] += amount  # Uncalled wager, never at risk.
            continue
        eligible = [i for i in paid if not folded[i]]
        if not eligible:
            raise ValueError('Pot has no eligible player')
        best = max(ranks[i] for i in eligible)
        winners = [i for i in eligible if ranks[i] == best]
        share, odd = divmod(amount, len(winners))
        for i in winners:
            awards[i] += share
        order = sorted(winners, key=lambda i: (i - button - 1) % n)
        for i in order[:odd]:
            awards[i] += 1
    return awards


class Hand:
    def __init__(self, stacks, button, sb, bb, rng=None, deck=None, hand_no=0,
                 payouts=None, evaluator=None):
        if bb <= 0 or sb < 0 or sb > bb or len(stacks) < 2:
            raise ValueError('Invalid blinds or table')
        if any(s < 0 or int(s) != s for s in stacks):
            raise ValueError('Stacks must be nonnegative integer chips')
        self.n = len(stacks)
        self.stacks = list(stacks)
        self.starting_stacks = list(stacks)
        self.total_chips = sum(stacks)
        self.dealt = [s > 0 for s in stacks]
        if sum(self.dealt) < 2 or not self.dealt[button]:
            raise ValueError('Need two live players and live button')
        self.button, self.sb, self.bb = button, sb, bb
        self.hand_no, self.payouts = hand_no, payouts
        self.evaluator = evaluator or evaluate_reference
        self.folded = [not x for x in self.dealt]
        self.all_in = [False] * self.n
        self.committed = [0] * self.n
        self.contributions = [0] * self.n
        self.history = []
        self.board = []
        self.street = 0
        self.done = False
        self.awards = None
        self.current_bet = bb
        self.last_full_raise = bb
        self.acted_bet = [None] * self.n
        self.pending = set()
        self.errors = self.timeouts = self.decisions = 0
        self.latencies_ns = []
        self.errors_by_seat = [0] * self.n
        self.timeouts_by_seat = [0] * self.n
        self.decisions_by_seat = [0] * self.n
        self.latencies_by_seat = [[] for _ in stacks]
        self.actions_by_seat = [dict(fold=0, check=0, call=0, raise_=0) for _ in stacks]
        self.deck = list(deck) if deck is not None else list(range(52))
        if deck is None:
            (rng or random.Random()).shuffle(self.deck)
        if len(self.deck) != 52 or len(set(self.deck)) != 52 or set(self.deck) != set(range(52)):
            raise ValueError('Deck must be a permutation of 0..51')
        self.holes = [[] for _ in stacks]
        order = [i for offset in range(1, self.n + 1)
                 if self.dealt[i := (button + offset) % self.n]]
        for _ in range(2):
            for i in order:
                self.holes[i].append(self.deck.pop())
        self.sb_seat = button if sum(self.dealt) == 2 else self.next_live(button)
        self.bb_seat = self.next_live(self.sb_seat)
        self._pay(self.sb_seat, min(sb, self.stacks[self.sb_seat]))
        self._pay(self.bb_seat, min(bb, self.stacks[self.bb_seat]))
        self.pending = set(self.actionable())
        self.actor = self._next_pending(self.bb_seat)
        self._advance_if_complete()

    def next_live(self, seat):
        for offset in range(1, self.n + 1):
            i = (seat + offset) % self.n
            if self.dealt[i]:
                return i
        raise RuntimeError('No live seat')

    def actionable(self):
        return [i for i in range(self.n) if not self.folded[i] and self.stacks[i] > 0]

    def _next_pending(self, seat):
        for offset in range(1, self.n + 1):
            i = (seat + offset) % self.n
            if i in self.pending:
                return i
        return None

    def _pay(self, seat, amount):
        self.stacks[seat] -= amount
        self.committed[seat] += amount
        self.contributions[seat] += amount
        self.all_in[seat] = self.stacks[seat] == 0

    def can_raise(self, seat):
        if self.folded[seat] or self.stacks[seat] <= max(0, self.current_bet - self.committed[seat]):
            return False
        if not any(i != seat and not self.folded[i]
                   and self.stacks[i] + self.committed[i] > self.current_bet
                   for i in range(self.n)):
            return False
        acted = self.acted_bet[seat]
        return acted is None or self.current_bet - acted >= self.last_full_raise

    def observation(self, seat=None):
        seat = self.actor if seat is None else seat
        if self.done or seat is None:
            raise ValueError('No decision pending')
        return dict(hero=seat, hole=list(self.holes[seat]), board=list(self.board),
                    stacks=list(self.stacks), committed=list(self.committed),
                    contributions=list(self.contributions), folded=list(self.folded),
                    all_in=list(self.all_in), button=self.button, street=self.street,
                    pot=sum(self.contributions),
                    to_call=min(self.stacks[seat], max(0, self.current_bet - self.committed[seat])),
                    min_raise_to=self.current_bet + self.last_full_raise,
                    max_raise_to=self.committed[seat] + self.stacks[seat],
                    can_raise=self.can_raise(seat), bb=self.bb, sb=self.sb,
                    hand_no=self.hand_no, payouts=self.payouts,
                    history=[dict(h) for h in self.history],
                    starting_stacks=list(self.starting_stacks),
                    current_bet=self.current_bet)

    def step(self, action):
        if self.done:
            raise ValueError('Hand is over')
        seat = self.actor
        if not isinstance(action, dict) or action.get('action') not in {'fold', 'check', 'call', 'raise'}:
            raise ValueError('Invalid action')
        kind = action['action']
        call = min(self.stacks[seat], max(0, self.current_bet - self.committed[seat]))
        amount = self.committed[seat]
        if kind == 'check' and call:
            raise ValueError('Cannot check facing a bet')
        if kind == 'raise':
            amount = action.get('amount')
            if isinstance(amount, bool) or not isinstance(amount, int):
                raise ValueError('Raise amount must be integer raise-to')
            maximum = self.committed[seat] + self.stacks[seat]
            minimum = self.current_bet + self.last_full_raise
            if not self.can_raise(seat) or amount <= self.current_bet or amount > maximum:
                raise ValueError('Raise not allowed or out of range')
            if amount < minimum and amount != maximum:
                raise ValueError('Under-minimum raise must be all-in')
        # Validation above precedes ALL mutation.
        self.pending.discard(seat)
        if kind == 'fold':
            self.folded[seat] = True
        elif kind == 'call':
            self._pay(seat, call)
            amount = self.committed[seat]
        elif kind == 'raise':
            increment = amount - self.current_bet
            self._pay(seat, amount - self.committed[seat])
            if increment >= self.last_full_raise:
                self.last_full_raise = increment
            self.current_bet = amount
            self.pending.update(i for i in self.actionable()
                                if i != seat and self.committed[i] < self.current_bet)
        self.acted_bet[seat] = self.current_bet
        self.history.append(dict(seat=seat, street=self.street, action=kind, amount=amount))
        self.actor = self._next_pending(seat)
        self._advance_if_complete()
        assert sum(self.stacks) + (0 if self.done else sum(self.contributions)) == self.total_chips

    def _advance_if_complete(self):
        remaining = [i for i in range(self.n) if not self.folded[i]]
        if len(remaining) == 1:
            self.awards = [0] * self.n
            self.awards[remaining[0]] = sum(self.contributions)
            self._finish()
            return
        actionable = self.actionable()
        # Do not ask a lone player to bet into opponents who are all-in.
        if len(actionable) <= 1:
            # A nominal full big blind need not be completed when EVERY other
            # opponent is all-in below it: no unmatched side pot can be bet.
            self.current_bet = max(self.committed[i] for i in remaining)
            if not actionable or self.committed[actionable[0]] >= self.current_bet:
                self.pending.clear()
        if self.pending:
            if self.actor not in self.pending:
                self.actor = self._next_pending(self.button)
            return
        if self.street == 3 or len(actionable) <= 1:
            while len(self.board) < 5:
                self.board.append(self.deck.pop())
            ranks = [self.evaluator(self.holes[i] + self.board) if not self.folded[i] else None
                     for i in range(self.n)]
            self.awards = settle_pots(self.contributions, self.folded, ranks, self.button)
            self._finish()
            return
        self.street += 1
        for _ in range(3 if self.street == 1 else 1):
            self.board.append(self.deck.pop())
        self.committed = [0] * self.n
        self.current_bet = 0
        self.last_full_raise = self.bb
        self.acted_bet = [None] * self.n
        self.pending = set(actionable)
        self.actor = self._next_pending(self.button)

    def _finish(self):
        self.stacks = [s + a for s, a in zip(self.stacks, self.awards)]
        self.done = True
        self.actor = None
        self.pending.clear()
        assert sum(self.stacks) == self.total_chips

    def play(self, agents, enforce_time_limit=False, record_latencies=False):
        """Play with exception/illegal/time fallback check, otherwise fold.

        Timing covers Agent.act only, not observation creation. It is local wall
        time, hence includes OS scheduling pauses. Disabled enforcement by default.
        """
        while not self.done:
            seat = self.actor
            obs = self.observation()
            agent = agents[self.actor]
            fallback = {'action': 'check' if obs['to_call'] == 0 else 'fold'}
            self.decisions += 1
            self.decisions_by_seat[seat] += 1
            try:
                start = time.perf_counter_ns()
                action = agent.act(obs)
                elapsed = time.perf_counter_ns() - start
                if record_latencies:
                    self.latencies_ns.append(elapsed)
                    self.latencies_by_seat[seat].append(elapsed)
                if enforce_time_limit and elapsed > 1_000_000:
                    self.timeouts += 1
                    self.timeouts_by_seat[seat] += 1
                    action = fallback
                self.step(action)
                key = 'raise_' if action['action'] == 'raise' else action['action']
                self.actions_by_seat[seat][key] += 1
            except Exception:
                self.errors += 1
                self.errors_by_seat[seat] += 1
                self.step(fallback)
                self.actions_by_seat[seat][fallback['action']] += 1
        return self.stacks


def tournament(agents, *, seed, payouts, blind_schedule, initial_stack=2000,
               max_hands=200, evaluator=None, enforce_time_limit=False,
               record_latencies=False):
    """Run one LOCAL sit-and-go. blind_schedule is [(sb,bb),...] per 25 hands.

    Payouts are descending rank order and synthetic/configured by caller. Bots
    should be freshly instantiated for each table, satisfying table state reset.
    A hand history is not retained after the hand; result includes diagnostics.
    """
    n = len(agents)
    if n < 2 or initial_stack <= 0 or int(initial_stack) != initial_stack or max_hands < 0:
        raise ValueError('Invalid table size, initial stack or hand limit')
    if len(payouts) != n or not blind_schedule or any(payouts[i] < payouts[i+1] for i in range(n-1)):
        raise ValueError('Need descending payouts and explicit blind schedule')
    rng = random.Random(seed)
    stacks = [initial_stack] * n
    points = [0.0] * n
    button = rng.randrange(n)
    errors = timeouts = decisions = 0
    latencies = []
    errors_by_seat = [0] * n
    timeouts_by_seat = [0] * n
    decisions_by_seat = [0] * n
    latencies_by_seat = [[] for _ in agents]
    actions_by_seat = [dict(fold=0, check=0, call=0, raise_=0) for _ in agents]
    finish_probabilities = [[0.0] * n for _ in agents]
    hands = 0
    for hand_no in range(max_hands):
        if sum(s > 0 for s in stacks) <= 1:
            break
        if stacks[button] == 0:
            button = next((button + k) % n for k in range(1, n+1) if stacks[(button + k) % n] > 0)
        before = list(stacks)
        sb, bb = blind_schedule[min(hand_no // 25, len(blind_schedule)-1)]
        hand = Hand(stacks, button, sb, bb, rng=rng, hand_no=hand_no,
                    payouts=payouts, evaluator=evaluator)
        stacks = hand.play(agents, enforce_time_limit, record_latencies)
        hands += 1
        errors += hand.errors
        timeouts += hand.timeouts
        decisions += hand.decisions
        latencies.extend(hand.latencies_ns)
        for i in range(n):
            errors_by_seat[i] += hand.errors_by_seat[i]
            timeouts_by_seat[i] += hand.timeouts_by_seat[i]
            decisions_by_seat[i] += hand.decisions_by_seat[i]
            latencies_by_seat[i].extend(hand.latencies_by_seat[i])
            for action in actions_by_seat[i]:
                actions_by_seat[i][action] += hand.actions_by_seat[i][action]
        bust = sorted((i for i in range(n) if before[i] > 0 and stacks[i] == 0), key=lambda i: before[i])
        rank = sum(s > 0 for s in before)
        while bust:
            tied = [i for i in bust if before[i] == before[bust[0]]]
            award = sum(payouts[rank-len(tied):rank]) / len(tied)
            for i in tied:
                points[i] = award
                for place in range(rank-len(tied), rank):
                    finish_probabilities[i][place] = 1.0 / len(tied)
                bust.remove(i)
            rank -= len(tied)
        button = next((button + k) % n for k in range(1, n+1) if stacks[(button + k) % n] > 0)
    live = [i for i, s in enumerate(stacks) if s > 0]
    live_points = icm([stacks[i] for i in live], payouts[:len(live)])
    for i, value in zip(live, live_points):
        points[i] = value
    for place in range(len(live)):
        probability = icm([stacks[i] for i in live],
                          [float(j == place) for j in range(len(live))])
        for i, value in zip(live, probability):
            finish_probabilities[i][place] = value
    return dict(stacks=stacks, points=points, hands=hands,
                errors=errors, timeouts=timeouts, decisions=decisions,
                latencies_ns=latencies, errors_by_seat=errors_by_seat,
                timeouts_by_seat=timeouts_by_seat, decisions_by_seat=decisions_by_seat,
                latencies_by_seat=latencies_by_seat, actions_by_seat=actions_by_seat,
                finish_probabilities=finish_probabilities)
