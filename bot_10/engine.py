"""Independent local no-limit Hold'em harness, not an official contest adapter.

Cards: integers rank*4+suit, ranks 0=2 through 12=A. Raise amounts are total
street commitments; calls use the incremental amount. No ante is assumed.
"""
from __future__ import annotations

from collections import defaultdict
import random
import time


class Hand:
    def __init__(self, agents, stacks, button=0, small_blind=10, big_blind=20,
                 seed=0, deck=None, hand_id=0, enforce_time=False):
        if len(agents) != len(stacks) or len(stacks) < 2:
            raise ValueError("agents and stacks must have equal length >=2")
        if any(type(x) is not int or x < 0 for x in stacks):
            raise ValueError("stacks must be nonnegative integers")
        if not 0 < small_blind <= big_blind:
            raise ValueError("invalid blinds")
        self.agents = agents
        self.initial = list(stacks)
        self.n = len(stacks)
        self.players = [dict(seat=i, stack=s, bet=0, committed=0,
                             folded=(s == 0), all_in=False) for i, s in enumerate(stacks)]
        self.active = [i for i, s in enumerate(stacks) if s > 0]
        if len(self.active) < 2:
            raise ValueError("at least two funded seats required")
        self.button = next((i for i in self.clockwise(button, include=True)
                            if i in self.active))
        self.sb = self.button if len(self.active) == 2 else self.next_active(self.button)
        self.bb = self.next_active(self.sb)
        self.small_blind, self.big_blind = int(small_blind), int(big_blind)
        self.hand_id, self.enforce_time = hand_id, enforce_time
        self.history, self.board = [], []
        self.holes = [[] for _ in stacks]
        self.errors, self.decisions_ns, self.payouts = [], [], [0] * self.n
        self.street = 0
        self.maxbet, self.last_full_raise = self.big_blind, self.big_blind
        self.last_action_bet = {}
        if deck is None:
            self.deck = list(range(52))
            random.Random(seed).shuffle(self.deck)
        else:
            self.deck = list(deck)
            if len(self.deck) != 52 or set(self.deck) != set(range(52)):
                raise ValueError("deck must be a permutation of 0..51")
        self.deck_index = 0
        deal_order = [i for i in self.clockwise(self.button) if i in self.active]
        for _ in range(2):
            for seat in deal_order:
                self.holes[seat].append(self.draw())
        self.commit(self.sb, min(self.small_blind, self.players[self.sb]['stack']))
        self.commit(self.bb, min(self.big_blind, self.players[self.bb]['stack']))
        self.history.append(dict(seat=self.sb, street=0, action='small_blind', amount=self.players[self.sb]['bet']))
        self.history.append(dict(seat=self.bb, street=0, action='big_blind', amount=self.players[self.bb]['bet']))

    def clockwise(self, seat, include=False):
        return [(seat + i) % self.n for i in range(0 if include else 1, self.n + (0 if include else 1))]

    def next_active(self, seat):
        return next(i for i in self.clockwise(seat) if i in self.active)

    def draw(self):
        c = self.deck[self.deck_index]
        self.deck_index += 1
        return c

    def live(self):
        return [i for i, p in enumerate(self.players) if not p['folded']]

    def actionable(self):
        return [i for i in self.live() if self.players[i]['stack'] > 0]

    def commit(self, seat, amount):
        p = self.players[seat]
        if amount < 0 or amount > p['stack']:
            raise ValueError("invalid chip commitment")
        p['stack'] -= amount
        p['bet'] += amount
        p['committed'] += amount
        p['all_in'] = p['stack'] == 0

    def can_raise(self, seat):
        p = self.players[seat]
        if p['folded'] or p['stack'] + p['bet'] <= self.maxbet:
            return False
        # No betting into a dry side pot against opponents already all-in.
        if not any(j != seat and self.players[j]['stack'] > 0 for j in self.live()):
            return False
        prior = self.last_action_bet.get(seat)
        return prior is None or self.maxbet - prior >= self.last_full_raise

    def observation(self, seat):
        p = self.players[seat]
        return dict(hole=list(self.holes[seat]), board=list(self.board), hero=seat,
                    button=self.button, small_blind=self.small_blind, big_blind=self.big_blind,
                    pot=sum(q['committed'] for q in self.players),
                    to_call=min(p['stack'], max(0, self.maxbet - p['bet'])),
                    min_raise_to=self.maxbet + self.last_full_raise,
                    max_raise_to=p['bet'] + p['stack'], can_raise=self.can_raise(seat),
                    players=[dict(q) for q in self.players], street=self.street,
                    hand_id=self.hand_id, history=[dict(a) for a in self.history])

    def apply_action(self, seat, action):
        """Apply legal action; reject illegal actions with ValueError (no silent repairs)."""
        p = self.players[seat]
        if p['folded'] or p['stack'] == 0:
            raise ValueError("inactive player cannot act")
        kind = action.get('action') if isinstance(action, dict) else None
        due = max(0, self.maxbet - p['bet'])
        full_raise = False
        if kind == 'fold':
            p['folded'] = True
            amount = 0
        elif kind == 'check':
            if due:
                raise ValueError("cannot check facing a bet")
            amount = 0
            # A check preserves the ability to raise a subsequent opening wager.
            self.last_action_bet[seat] = None if self.maxbet == 0 else self.maxbet
        elif kind == 'call':
            amount = min(due, p['stack'])
            self.commit(seat, amount)
            self.last_action_bet[seat] = self.maxbet
        elif kind == 'raise':
            target = action.get('amount')
            if type(target) is not int or not self.can_raise(seat):
                raise ValueError("raise unavailable or noninteger")
            ceiling = p['bet'] + p['stack']
            if target <= self.maxbet or target > ceiling:
                raise ValueError("raise outside stack bounds")
            delta = target - self.maxbet
            if delta < self.last_full_raise and target != ceiling:
                raise ValueError("under-minimum raise requires all-in")
            self.commit(seat, target - p['bet'])
            amount = target
            full_raise = delta >= self.last_full_raise
            if full_raise:
                self.last_full_raise = delta
            self.maxbet = target
            self.last_action_bet[seat] = target
        else:
            raise ValueError("unknown action")
        self.history.append(dict(seat=seat, street=self.street, action=kind, amount=amount))
        return full_raise

    def betting_round(self, start):
        pending = set(self.actionable())
        seat = start
        actions = 0
        while pending and len(self.live()) > 1:
            actors = self.actionable()
            if len(actors) == 1:
                # A short big blind sets a nominal bring-in while multiple
                # players can bet. Once everyone else is all-in, only chips
                # actually wagered by a live opponent can be called.
                self.maxbet = max(self.players[i]['bet'] for i in self.live())
                if self.players[actors[0]]['bet'] >= self.maxbet:
                    break
            candidates = [i for i in self.clockwise(seat, include=True) if i in pending and i in actors]
            if not candidates:
                break
            seat = candidates[0]
            obs = self.observation(seat)
            before = time.perf_counter_ns()
            try:
                agent = self.agents[seat]
                action = agent.act(obs) if hasattr(agent, 'act') else agent(obs)
                duration = time.perf_counter_ns() - before
                self.decisions_ns.append(duration)
                if self.enforce_time and duration > 1_000_000:
                    raise TimeoutError("decision exceeded 1ms")
                oldmax = self.maxbet
                full = self.apply_action(seat, action)
            except Exception as exc:
                self.errors.append(dict(seat=seat, street=self.street, error=repr(exc)))
                oldmax = self.maxbet
                full = self.apply_action(seat, dict(action='check' if not obs['to_call'] else 'fold'))
            pending.discard(seat)
            if full:
                pending = set(self.actionable()) - {seat}
            elif self.maxbet > oldmax:
                pending.update(i for i in self.actionable() if i != seat and self.players[i]['bet'] < self.maxbet)
            pending.intersection_update(self.actionable())
            seat = (seat + 1) % self.n
            actions += 1
            if actions > 10000:
                raise RuntimeError("betting did not converge")
        self.refund_uncalled()

    def refund_uncalled(self):
        # Return the unmatched portion before later-street pot observations.
        bets = sorted((p['bet'], i) for i, p in enumerate(self.players))
        highest, seat = bets[-1]
        second = bets[-2][0]
        if highest > second:
            p = self.players[seat]
            refund = highest - second
            p['stack'] += refund
            p['bet'] -= refund
            p['committed'] -= refund
            p['all_in'] = p['stack'] == 0

    def settle(self):
        live = self.live()
        total = sum(p['committed'] for p in self.players)
        if len(live) == 1:
            self.payouts[live[0]] = total
        else:
            from cards import evaluate
            scores = {i: evaluate(self.holes[i] + self.board) for i in live}
            levels = sorted({p['committed'] for p in self.players if p['committed'] > 0})
            prior = 0
            for level in levels:
                contributors = [i for i, p in enumerate(self.players) if p['committed'] >= level]
                amount = (level - prior) * len(contributors)
                eligible = [i for i in contributors if i in scores]
                if not eligible:
                    # An unmatched overbet by a folder cannot arise in legal play.
                    raise RuntimeError("pot layer without a live claimant")
                best = max(scores[i] for i in eligible)
                winners = [i for i in self.clockwise(self.button) if i in eligible and scores[i] == best]
                share, odd = divmod(amount, len(winners))
                for index, i in enumerate(winners):
                    self.payouts[i] += share + (index < odd)
                prior = level
        for i, p in enumerate(self.players):
            p['stack'] += self.payouts[i]
        assert sum(p['stack'] for p in self.players) == sum(self.initial)

    def run(self):
        self.betting_round(self.next_active(self.bb))
        for street, count in ((1, 3), (2, 1), (3, 1)):
            if len(self.live()) <= 1:
                break
            self.street = street
            self.draw()  # burn
            self.board.extend(self.draw() for _ in range(count))
            for p in self.players:
                p['bet'] = 0
            self.maxbet, self.last_full_raise = 0, self.big_blind
            self.last_action_bet = {}
            self.betting_round(self.next_active(self.button))
        self.settle()
        final = [p['stack'] for p in self.players]
        return dict(stacks=final, history=self.history, board=self.board, holes=self.holes,
                    payouts=self.payouts, errors=self.errors, decisions_ns=self.decisions_ns,
                    eliminations=[i for i in self.active if final[i] == 0])


def play_hand(agents, stacks, button=0, small_blind=10, big_blind=20,
              seed=0, deck=None, hand_id=0, enforce_time=False):
    return Hand(agents, stacks, button, small_blind, big_blind, seed, deck,
                hand_id, enforce_time).run()
