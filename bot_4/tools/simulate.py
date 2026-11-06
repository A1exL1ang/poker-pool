"""Independent local NLHE test harness; not an official contest simulator.

Includes full raises, short all-ins/reopening, side pots, blind rotation,
heads-up rules, showdown ties, odd chips and exact ICM at the hand cap.
Schedules and scoring are explicitly illustrative/configurable.
"""
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent import Agent, State, Player, Event, Action, legalize
from bot.cards import evaluate, features, preflop_index
from bot.model import PREFLOP, predict
from bot.icm import icm


@dataclass
class Seat:
    stack: int
    committed: int = 0
    contributed: int = 0
    folded: bool = False
    hole: tuple = ()


def distribute(contributions, folded, ranks, odd_order):
    """Return complete pot awards, including unmatched bet refunds."""
    awards = [0] * len(contributions)
    previous = 0
    for cap in sorted(set(c for c in contributions if c > 0)):
        participants = [i for i, c in enumerate(contributions) if c >= cap]
        amount = (cap - previous) * len(participants)
        eligible = [i for i in participants if not folded[i]]
        if not eligible:
            # A valid betting engine never creates an uncontested layer
            # funded solely by folded players above every live contribution.
            raise AssertionError("pot layer has no eligible player")
        best = max(ranks[i] for i in eligible)
        winners = [i for i in odd_order if i in eligible and ranks[i] == best]
        share, odd = divmod(amount, len(winners))
        for j, i in enumerate(winners):
            awards[i] += share + (j < odd)
        previous = cap
    assert sum(awards) == sum(contributions)
    return awards


class Baseline:
    """Simple local opponents for validation, not competition submissions."""
    def __init__(self, kind, seed):
        self.kind = kind
        self.rng = random.Random(seed)
        self.cache_key = self.cache = None

    def decide(self, s):
        h = next(p for p in s.players if p.seat == s.hero)
        call = min(s.to_call, h.stack)
        n = sum(not p.folded and p.seat != s.hero for p in s.players)
        price = call / max(1, s.pot + call)
        if self.kind == "random":
            r = self.rng.random()
            a = Action("raise", h.committed + call + max(s.big_blind, s.pot // 2)) if r < .10 else Action("call" if call else "check") if r < .78 else Action("fold")
            return legalize(s, a)
        if self.kind == "station":
            if not s.board:
                e = PREFLOP["random"][preflop_index(s.hole)]
            else:
                e = predict(features(s.hole, s.board))[1]
            if e > .87:
                return legalize(s, Action("raise", h.committed + call + s.pot))
            if not call:
                return Action("check")
            return Action("call" if call else "check") if e > price * .78 else Action("fold")
        if not s.board:
            ix = preflop_index(s.hole)
            q = PREFLOP["percentile"][ix]
            limit = .16 if self.kind == "tight" else .32
            if q < .045 or (q < limit and s.to_call <= s.big_blind):
                return legalize(s, Action("raise", max(s.min_raise_to, 3 * s.big_blind)))
            if not call:
                return Action("check")
            return Action("call") if q < limit and call <= h.stack * .30 else Action("fold")
        key = (s.hole, s.board)
        if key != self.cache_key:
            self.cache_key, self.cache = key, predict(features(s.hole, s.board))
        e = self.cache[1] ** max(1, n * .85)
        if self.kind == "tight":
            e -= .10
        if e > .69:
            return legalize(s, Action("raise", h.committed + call + max(s.big_blind, int(s.pot * .65))))
        if not call:
            return Action("check")
        return Action("call") if e > price + .04 else Action("fold")


class Table:
    def __init__(self, agents, seed=0, stacks=None, schedule=(20, 30, 50, 80, 120, 200, 300, 500), payouts=(), hand_cap=200, capture=None):
        self.agents = agents
        self.n = len(agents)
        self.seats = [Seat(s) for s in (stacks or [2000] * self.n)]
        self.total = sum(p.stack for p in self.seats)
        self.rng = random.Random(seed)
        self.table_id = f"local-{seed}"
        self.button = seed % self.n
        self.schedule = schedule
        self.payouts = tuple(payouts)
        self.hand_cap = hand_cap
        self.results = [None] * self.n
        self.hand_id = 0
        self.actions = Counter()
        self.capture = capture
        self.decision_ns = []

    def clockwise(self, start, seats):
        return sorted(seats, key=lambda x: (x - start - 1) % self.n)

    def pay(self, i, amount):
        p = self.seats[i]
        paid = min(p.stack, max(0, amount))
        p.stack -= paid
        p.committed += paid
        p.contributed += paid
        return paid

    def state(self, i, bb, current, increment, can_raise):
        p = self.seats[i]
        return State(self.table_id, self.hand_id, i, self.button, p.hole, tuple(self.board),
                     tuple(Player(j, q.stack, q.committed, q.contributed, q.folded, q.stack == 0)
                           for j, q in enumerate(self.seats) if j in self.dealt),
                     sum(q.contributed for q in self.seats), bb, max(0, current - p.committed),
                     current + increment, p.committed + p.stack, can_raise,
                     tuple(self.history), self.payouts)

    def betting(self, bb, first, preflop=False):
        current = bb if preflop else 0
        increment = bb
        pending = {i for i in self.dealt if not self.seats[i].folded and self.seats[i].stack > 0}
        acted_at = {}
        cursor = (first - 1) % self.n
        guard = 0
        while pending:
            live = [i for i in self.dealt if not self.seats[i].folded]
            if len(live) <= 1:
                return
            able = [i for i in live if self.seats[i].stack > 0]
            if len(able) == 1 and self.seats[able[0]].committed >= current:
                return
            i = self.clockwise(cursor, pending)[0]
            pending.remove(i)
            cursor = i
            p = self.seats[i]
            reopened = i not in acted_at or current - acted_at[i] >= increment
            can_raise = reopened and any(j != i and self.seats[j].stack > 0 for j in live)
            s = self.state(i, bb, current, increment, can_raise)
            tick = time.perf_counter_ns()
            action = self.agents[i].decide(s)
            elapsed = time.perf_counter_ns() - tick
            if isinstance(self.agents[i], Agent):
                self.decision_ns.append(elapsed)
            legal = legalize(s, action)
            if legal != action:
                raise AssertionError(f"illegal action {action}; normalized {legal}; {s}")
            if self.capture is not None and isinstance(self.agents[i], Agent) and len(self.capture) < 3000:
                self.capture.append(s)
            self.actions[action.action] += 1
            if action.action == "fold":
                p.folded = True
            elif action.action == "call":
                self.pay(i, s.to_call)
            elif action.action == "raise":
                self.pay(i, action.amount - p.committed)
                change = p.committed - current
                if change >= increment:
                    increment = change
                current = p.committed
                pending.update(j for j in live if j != i and self.seats[j].stack > 0 and self.seats[j].committed < current)
            acted_at[i] = current
            self.history.append(Event(i, s.street, action.action, p.committed, s.to_call, s.pot))
            guard += 1
            if guard > 300:
                raise AssertionError("betting loop did not converge")

    def hand(self):
        self.dealt = [i for i, p in enumerate(self.seats) if p.stack > 0]
        if self.button not in self.dealt:
            self.button = self.clockwise(self.button, self.dealt)[0]
        previous_stacks = [p.stack for p in self.seats]
        deck = list(range(52))
        self.rng.shuffle(deck)
        for i, p in enumerate(self.seats):
            p.committed = p.contributed = 0
            p.folded = i not in self.dealt
            p.hole = (deck.pop(), deck.pop()) if i in self.dealt else ()
        self.board = []
        self.history = []
        bb = self.schedule[min(len(self.schedule) - 1, self.hand_id // 25)]
        order = self.clockwise(self.button, self.dealt)
        sbseat = self.button if len(self.dealt) == 2 else order[0]
        bbseat = order[0] if len(self.dealt) == 2 else order[1]
        self.pay(sbseat, bb // 2)
        self.pay(bbseat, bb)
        self.betting(bb, self.clockwise(bbseat, self.dealt)[0], True)
        for street_cards in (3, 1, 1):
            if sum(not self.seats[i].folded for i in self.dealt) <= 1:
                break
            self.board.extend(deck.pop() for _ in range(street_cards))
            for p in self.seats:
                p.committed = 0
            self.betting(bb, self.clockwise(self.button, self.dealt)[0])
        live = [i for i in self.dealt if not self.seats[i].folded]
        # Return unmatched top contribution before making pot layers. This
        # matters when a player wins a fold with less total contribution than
        # an opponent who previously folded (e.g. short-stack side pots).
        contributions = [p.contributed for p in self.seats]
        ranked = sorted(range(self.n), key=lambda i: contributions[i], reverse=True)
        first, second = ranked[:2]
        if contributions[first] > contributions[second]:
            refund = contributions[first] - contributions[second]
            contributions[first] -= refund
            self.seats[first].stack += refund
            self.seats[first].contributed -= refund
        if len(live) == 1:
            awards = [0] * self.n
            awards[live[0]] = sum(contributions)
        else:
            ranks = [evaluate((*p.hole, *self.board)) if not p.folded else -1 for p in self.seats]
            awards = distribute(contributions, [p.folded for p in self.seats], ranks, self.clockwise(self.button, range(self.n)))
        # Feed only public history; no opponent hole cards enter observations.
        for i in self.dealt:
            if hasattr(self.agents[i], "observe"):
                self.agents[i].observe(self.state(i, bb, max(p.committed for p in self.seats), bb, False))
        for i, value in enumerate(awards):
            self.seats[i].stack += value
        assert sum(p.stack for p in self.seats) == self.total
        eliminated = [i for i in self.dealt if self.seats[i].stack == 0]
        # Simultaneous busts: larger start-of-hand stack finishes higher;
        # equal stacks split the affected points (explicit local assumption).
        left = len(self.dealt)
        for stack in sorted(set(previous_stacks[i] for i in eliminated)):
            group = [i for i in eliminated if previous_stacks[i] == stack]
            points = [self.payouts[j - 1] if j <= len(self.payouts) else 0. for j in range(left - len(group) + 1, left + 1)]
            for i in group:
                self.results[i] = sum(points) / len(group)
            left -= len(group)
        survivors = [i for i, p in enumerate(self.seats) if p.stack > 0]
        if survivors:
            self.button = self.clockwise(self.button, survivors)[0]
        self.hand_id += 1

    def run(self):
        while self.hand_id < self.hand_cap and sum(p.stack > 0 for p in self.seats) > 1:
            self.hand()
        survivors = [i for i, p in enumerate(self.seats) if p.stack > 0]
        values = icm([self.seats[i].stack for i in survivors], self.payouts)
        for i, value in zip(survivors, values):
            self.results[i] = value
        assert abs(sum(self.results) - sum(self.payouts[:self.n])) < 1e-6
        return self.results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4100)
    ap.add_argument("--field", choices=("mixed", "tight", "station", "equity", "self"), default="mixed")
    ap.add_argument("--payout", choices=("wta", "top3"), default="top3")
    ap.add_argument("--output", default="evidence/simulation.json")
    ap.add_argument("--capture", action="store_true")
    args = ap.parse_args()
    pay = (1., 0., 0., 0., 0., 0., 0., 0.) if args.payout == "wta" else (.5, .3, .2, 0., 0., 0., 0., 0.)
    scores, chips, wins, times = [], [], 0, []
    actions = Counter()
    capture = [] if args.capture else None
    start = time.perf_counter()
    for t in range(args.tables):
        seat = t % 8
        agents = []
        for i in range(8):
            kind = ("tight", "equity", "station", "equity", "tight", "random", "equity")[i % 7] if args.field == "mixed" else args.field
            agents.append(Agent(404 + i) if i == seat or kind == "self" else Baseline(kind, args.seed + t * 8 + i))
        table = Table(agents, args.seed + t, payouts=pay, capture=capture)
        result = table.run()
        scores.append(result[seat])
        chips.append(table.seats[seat].stack)
        wins += table.seats[seat].stack == table.total
        times.extend(table.decision_ns)
        actions.update(table.actions)
        if (t + 1) % 20 == 0:
            print(f"{t+1}/{args.tables} tables; mean points {sum(scores)/len(scores):.5f}", flush=True)
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / max(1, len(scores) - 1)
    times.sort()
    report = {"tables": args.tables, "seed": args.seed, "field": args.field, "payouts": pay,
              "blind_schedule": [20, 30, 50, 80, 120, 200, 300, 500], "schedule_is_illustrative": True,
              "points_mean": mean, "points_standard_error": (variance / len(scores)) ** .5,
              "equal_share_points": 1/8, "outright_wins": wins,
              "final_chips_mean": sum(chips) / len(chips), "seconds": time.perf_counter() - start,
              "decisions": len(times), "decision_median_us": times[len(times)//2]/1000,
              "decision_p99_us": times[int(len(times)*.99)]/1000,
              "decision_max_us": max(times)/1000, "decisions_over_1ms": sum(x>1000000 for x in times),
              "all_seat_actions": actions, "scores": scores}
    Path(args.output).write_text(json.dumps(report, indent=2))
    if capture is not None:
        with (ROOT / "evidence/observations.jsonl").open("w") as f:
            for s in capture:
                f.write(json.dumps(asdict(s), separators=(",", ":")) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "scores"}, indent=2))


if __name__ == "__main__":
    main()
