"""Original local NLHE harness, NOT an official contest simulator.

Includes heads-up blind order, full/short raise reopening, all-ins, side pots,
odd chips, simultaneous eliminations and optional ICM hand-cap scoring.
All later blind levels and point schedules are explicit local experiments.
"""

import random
from dataclasses import dataclass

from kestrel.cards import evaluate
from kestrel.icm import value as icm_value
from kestrel.model import Action, Event, Observation, Player


LAB_BLINDS = ((10, 20), (15, 30), (25, 50), (40, 80),
              (60, 120), (100, 200), (150, 300), (250, 500))
LAB_POINTS = (8., 5., 3., 2., 1., 0., 0., 0.)


@dataclass(slots=True)
class Seat:
    stack: int
    committed: int = 0
    invested: int = 0
    folded: bool = False
    all_in: bool = False


def split_pots(contributions, folded, ranks, button, seat_order):
    """Return winnings by seat; this also returns unmatched contributions."""
    won = {seat: 0 for seat in contributions}
    previous = 0
    for level in sorted(set(contributions.values()) - {0}):
        contributors = [s for s, amount in contributions.items() if amount >= level]
        pot = (level - previous) * len(contributors)
        previous = level
        eligible = [s for s in contributors if s not in folded]
        if not eligible:
            raise AssertionError("Side pot without eligible player")
        best = max(ranks[s] for s in eligible)
        winners = [s for s in eligible if ranks[s] == best]
        base, odd = divmod(pot, len(winners))
        for s in winners:
            won[s] += base
        clockwise = [s for s in seat_order if s > button] + [s for s in seat_order if s <= button]
        for s in (s for s in clockwise if s in winners):
            if odd <= 0:
                break
            won[s] += 1
            odd -= 1
    assert sum(won.values()) == sum(contributions.values())
    return won


def action_is_legal(obs, action):
    if not isinstance(action, Action):
        return False
    if action.action == "fold":
        return obs.to_call > 0 and action.amount == 0
    if action.action == "check":
        return obs.to_call == 0 and action.amount == 0
    if action.action == "call":
        return obs.to_call > 0 and action.amount == 0
    if action.action != "raise" or not obs.can_raise or type(action.amount) is not int:
        return False
    h = next(p for p in obs.players if p.seat == obs.hero)
    return (h.committed + obs.to_call < action.amount <= obs.max_raise_to
            and (action.amount >= obs.min_raise_to or action.amount == h.stack + h.committed))


class Table:
    def __init__(self, agents, seed=1, stacks=None, points=LAB_POINTS, blinds=LAB_BLINDS,
                 hand_cap=200, capture=None, table_id=None):
        self.agents = agents
        self.n = len(agents)
        self.rng = random.Random(seed)
        self.seats = {s: Seat((stacks or [2000] * self.n)[s]) for s in range(self.n)}
        self.total_chips = sum(p.stack for p in self.seats.values())
        self.points = tuple(points[:self.n])
        if len(self.points) != self.n:
            raise ValueError("Provide one local payout per player")
        self.blinds = blinds
        self.hand_cap = hand_cap
        self.capture = capture
        self.table_id = table_id or f"lab-{seed}"
        self.button = self.rng.randrange(self.n)
        self.hand_id = 0
        self.actions = 0
        self.awards = [None] * self.n
        self.finishes = [None] * self.n
        self.last_history = ()
        for a in self.agents:
            if hasattr(a, "reset"):
                a.reset(self.table_id)

    def _pay(self, seat, amount):
        p = self.seats[seat]
        paid = min(p.stack, amount)
        p.stack -= paid
        p.committed += paid
        p.invested += paid
        p.all_in = p.stack == 0
        return paid

    def _next(self, seat, eligible):
        for delta in range(1, self.n + 1):
            s = (seat + delta) % self.n
            if s in eligible:
                return s
        raise AssertionError("No eligible seat")

    def _observation(self, hero, current, min_increment, history, can_raise):
        p = self.seats[hero]
        return Observation(
            hand_id=self.hand_id, hero=hero, button=self.button,
            hole=self.holes[hero], board=tuple(self.board),
            players=tuple(Player(s, self.seats[s].stack, self.seats[s].committed,
                                 self.seats[s].invested, self.seats[s].folded,
                                 self.seats[s].all_in) for s in self.alive),
            pot=sum(self.seats[s].invested for s in self.alive),
            to_call=max(0, current - p.committed), min_raise_to=current + min_increment,
            max_raise_to=p.stack + p.committed, big_blind=self.bb,
            can_raise=can_raise, history=tuple(history), payouts=self.points[:len(self.alive)],
            table_id=self.table_id, hands_remaining=self.hand_cap - self.hand_id)

    def _round(self, first, history, preflop=False):
        current = max(self.seats[s].committed for s in self.alive)
        # The full big blind remains the bring-in if the BB is all-in short.
        if preflop:
            current = max(current, self.bb)
        increment = self.bb
        eligible = {s for s in self.alive if not self.seats[s].folded and not self.seats[s].all_in}
        pending = set(eligible)
        last_acted_at = {}
        actor = first
        loop = 0
        while pending:
            live = [s for s in self.alive if not self.seats[s].folded]
            if len(live) <= 1:
                return
            eligible = {s for s in live if not self.seats[s].all_in}
            pending &= eligible
            if not pending:
                return
            if len(eligible) == 1:
                only = next(iter(eligible))
                opposing = max(self.seats[s].committed for s in live if s != only)
                # Never force a call to a nominal blind nobody can win, and no
                # dry side-pot betting when every opponent is already all-in.
                current = min(current, max(opposing, self.seats[only].committed))
                if self.seats[only].committed >= current:
                    return
            if actor not in pending:
                actor = self._next((actor - 1) % self.n, pending)
            p = self.seats[actor]
            reopened = actor not in last_acted_at or current - last_acted_at[actor] >= increment
            can_raise = reopened and len(eligible) > 1 and p.stack + p.committed > current
            obs = self._observation(actor, current, increment, history, can_raise)
            action = self.agents[actor].act(obs)
            if not action_is_legal(obs, action):
                raise AssertionError(f"Illegal action {action} for {obs}")
            if self.capture is not None:
                self.capture(obs, action)
            self.actions += 1
            pending.discard(actor)
            old = current
            if action.action == "fold":
                p.folded = True
                amount = 0
            elif action.action == "call":
                amount = self._pay(actor, obs.to_call)
            elif action.action == "raise":
                amount = action.amount
                self._pay(actor, amount - p.committed)
                current = amount
                if current - old >= increment:
                    increment = current - old
                pending |= {s for s in eligible if s != actor and self.seats[s].committed < current}
            else:
                amount = 0
            last_acted_at[actor] = current
            history.append(Event(len(history), actor, obs.street, action.action, amount, obs.pot, obs.to_call))
            loop += 1
            if loop > 1000:
                raise AssertionError("Betting loop failed to terminate")
            if pending:
                actor = self._next(actor, pending)

    def play_hand(self):
        self.alive = [s for s, p in self.seats.items() if p.stack > 0]
        if len(self.alive) <= 1:
            return
        self.button = self._next(self.button, self.alive)
        start_stacks = {s: self.seats[s].stack for s in self.alive}
        for s in self.alive:
            p = self.seats[s]
            p.committed = p.invested = 0
            p.folded = p.all_in = False
        self.sb, self.bb = self.blinds[min(len(self.blinds) - 1, self.hand_id // 25)]
        deck = list(range(52))
        self.rng.shuffle(deck)
        self.holes = {s: (deck.pop(), deck.pop()) for s in self.alive}
        self.board = []
        history = []
        sb = self.button if len(self.alive) == 2 else self._next(self.button, self.alive)
        bb = self._next(sb, self.alive)
        self._pay(sb, self.sb)
        self._pay(bb, self.bb)
        first = sb if len(self.alive) == 2 else self._next(bb, self.alive)
        self._round(first, history, preflop=True)
        for count in (3, 1, 1):
            live = [s for s in self.alive if not self.seats[s].folded]
            if len(live) <= 1:
                break
            deck.pop()  # burn
            self.board.extend(deck.pop() for _ in range(count))
            for s in self.alive:
                self.seats[s].committed = 0
            self._round(self._next(self.button, live), history)
        live = [s for s in self.alive if not self.seats[s].folded]
        contributions = {s: self.seats[s].invested for s in self.alive}
        if len(live) == 1:
            self.seats[live[0]].stack += sum(contributions.values())
        else:
            ranks = {s: evaluate(self.holes[s] + tuple(self.board)) for s in live}
            winnings = split_pots(contributions, set(self.alive) - set(live), ranks, self.button, self.alive)
            for s, chips in winnings.items():
                self.seats[s].stack += chips
        if sum(p.stack for p in self.seats.values()) != self.total_chips:
            raise AssertionError("Chips were created or destroyed")
        busted = sorted((s for s in self.alive if self.seats[s].stack == 0), key=start_stacks.__getitem__)
        remaining = len(self.alive)
        while busted:
            first = busted[0]
            tied = [s for s in busted if start_stacks[s] == start_stacks[first]]
            positions = list(range(remaining - len(tied), remaining))
            award = sum(self.points[p] for p in positions) / len(tied)
            finish = sum(p + 1 for p in positions) / len(tied)
            for s in tied:
                self.awards[s] = award
                self.finishes[s] = finish
                busted.remove(s)
            remaining -= len(tied)
        self.hand_id += 1
        self.last_history = tuple(history)

    def run(self):
        while self.hand_id < self.hand_cap and sum(p.stack > 0 for p in self.seats.values()) > 1:
            self.play_hand()
        alive = [s for s, p in self.seats.items() if p.stack > 0]
        stacks = [self.seats[s].stack for s in alive]
        for j, s in enumerate(alive):
            self.awards[s] = icm_value(stacks, j, self.points[:len(alive)])
            self.finishes[s] = 1.0 if len(alive) == 1 else None
        assert abs(sum(self.awards) - sum(self.points)) < 1e-7
        return {"points": self.awards, "stacks": [self.seats[s].stack for s in range(self.n)],
                "finishes": self.finishes, "hands": self.hand_id, "actions": self.actions}
