"""Local, self-authored standard NLHE simulator. Not the official contest engine.

Correctness priorities: integer chips, all-in underraises and reopening, uncalled
returns, side pots, split pots/odd chips, heads-up button order, conservation.
The illustrative blind progression doubles every 25 hands; only the initial
10/20 and 25-hand interval come from RULES.txt.
"""
import random
import time
from collections import Counter

from bot3.cards import evaluate
from bot3.interface import validate_action


def award_pots(players, board, button, holes):
    """Award tiered side pots. Return allocations; mutate players' stacks."""
    levels = sorted({p["committed"] for p in players if p["committed"] > 0})
    awards = [0] * len(players)
    values = {i: evaluate((*holes[i], *board)) for i, p in enumerate(players) if not p["folded"]}
    prev = 0
    for level in levels:
        contributors = [i for i, p in enumerate(players) if p["committed"] >= level]
        amount = (level - prev) * len(contributors)
        eligible = [i for i in contributors if not players[i]["folded"]]
        if not eligible:
            # The only such layer can be unmatched folded chips in malformed input.
            raise AssertionError("pot layer has no eligible player")
        top = max(values[i] for i in eligible)
        winners = [i for i in eligible if values[i] == top]
        winners.sort(key=lambda i: (i - button - 1) % len(players))
        each, odd = divmod(amount, len(winners))
        for j, i in enumerate(winners):
            awards[i] += each + (j < odd)
        prev = level
    for i, amount in enumerate(awards):
        players[i]["stack"] += amount
    return awards


class Table:
    def __init__(self, agents, seed=3, starting_stack=2000, max_hands=200, payouts=None, track_latency=False, table_id=None, decision_budget_ns=None):
        self.agents = agents
        self.rng = random.Random(seed)
        self.players = [{"seat": i, "stack": starting_stack, "bet": 0, "committed": 0, "folded": False, "all_in": False} for i in range(len(agents))]
        self.total = starting_stack * len(agents)
        self.button = self.rng.randrange(len(agents))
        self.max_hands = max_hands
        self.payouts = payouts
        self.track_latency = track_latency
        self.decision_budget_ns = decision_budget_ns
        self.timeouts = [0] * len(agents)
        self.table_id = table_id if table_id is not None else f"local-{seed}"
        self.latencies = [[] for _ in agents]
        self.actions = [Counter() for _ in agents]
        self.elimination_groups = []
        self.errors = []
        self.hands_played = 0
        self.showdowns = 0
        self.history = []
        for a in agents:
            if hasattr(a, "reset"):
                a.reset(self.table_id)

    def next_live(self, seat, live):
        n = len(self.players)
        for j in range(1, n + 1):
            k = (seat + j) % n
            if k in live:
                return k
        raise AssertionError("no live player")

    def pay(self, seat, amount):
        p = self.players[seat]
        amount = min(p["stack"], max(0, amount))
        p["stack"] -= amount
        p["bet"] += amount
        p["committed"] += amount
        p["all_in"] = p["stack"] == 0

    def observation(self, seat, board, street, bb, current, last_raise, can_raise):
        p = self.players[seat]
        call = min(p["stack"], max(0, current - p["bet"]))
        legal = ["fold", "call"] if call else ["check"]
        maximum = p["bet"] + p["stack"]
        if can_raise and maximum > current and any(q["stack"] > 0 and not q["folded"] and q["seat"] != seat for q in self.players):
            legal.append("raise")
        return {"table_id": self.table_id, "hand_id": self.hands_played, "hero_seat": seat, "button": self.button,
                "hole_cards": self.holes[seat], "board": board[:], "street": street,
                "players": [dict(q) for q in self.players], "pot": sum(q["committed"] for q in self.players),
                "big_blind": bb, "to_call": call, "current_bet": current, "min_raise_to": current + last_raise,
                "max_raise_to": maximum, "legal_actions": legal, "history": [dict(e) for e in self.history],
                **({"payouts": self.payouts} if self.payouts is not None else {})}

    def betting(self, start, board, street, bb, current):
        ps = self.players
        last_raise = bb
        pending = {i for i, p in enumerate(ps) if not p["folded"] and p["stack"] > 0}
        # Track bet level at last action: cumulative underraises reopen action
        # once their total increase reaches the last full raise size.
        acted_at = {}
        seat = start
        decisions = 0
        while pending:
            if sum(not p["folded"] for p in ps) <= 1:
                return
            if seat not in pending:
                seat = self.next_live(seat, pending)
            p = ps[seat]
            others_can_act = sum(q["stack"] > 0 and not q["folded"] for q in ps if q["seat"] != seat)
            if not others_can_act:
                # A nominal full big blind need not be matched when the only
                # opponent is already all-in for less than that amount.
                current = min(current, max(q["bet"] for q in ps if not q["folded"]))
            if not others_can_act and p["bet"] >= current:
                pending.remove(seat)
                continue
            o = self.observation(seat, board, street, bb, current, last_raise, seat not in acted_at or current - acted_at[seat] >= last_raise)
            timed = self.track_latency or self.decision_budget_ns is not None
            before = time.perf_counter_ns() if timed else 0
            action = self.agents[seat].act(o)
            elapsed = time.perf_counter_ns() - before if timed else 0
            if self.track_latency:
                self.latencies[seat].append(elapsed)
            validate_action(o, action)
            if self.decision_budget_ns is not None and elapsed > self.decision_budget_ns:
                self.timeouts[seat] += 1
                action = {"action": "check" if "check" in o["legal_actions"] else "fold", "amount": 0}
            name, amount = action["action"], action["amount"]
            self.actions[seat][name] += 1
            pending.discard(seat)
            previous = current
            pot_before = o["pot"]
            if name == "fold":
                p["folded"] = True
            elif name == "call":
                self.pay(seat, amount)
            elif name == "raise":
                self.pay(seat, amount - p["bet"])
                current = p["bet"]
                full = current - previous >= last_raise
                if full:
                    last_raise = current - previous
                    pending = {i for i, q in enumerate(ps) if i != seat and not q["folded"] and q["stack"] > 0}
                else:
                    pending |= {i for i, q in enumerate(ps) if i != seat and not q["folded"] and q["stack"] > 0 and q["bet"] < current}
            acted_at[seat] = current
            self.history.append({"seat": seat, "street": street, "action": name, "amount": amount, "pot_before": pot_before})
            decisions += 1
            if decisions > 300:
                raise AssertionError("betting round did not terminate")
            if pending:
                seat = self.next_live(seat, pending)

    def hand(self):
        ps = self.players
        alive = {i for i, p in enumerate(ps) if p["stack"] > 0}
        if len(alive) <= 1:
            return False
        before_stacks = [p["stack"] for p in ps]
        self.button = self.next_live(self.button, alive)
        sb = self.button if len(alive) == 2 else self.next_live(self.button, alive)
        big = self.next_live(sb, alive)
        # Illustrative fixture, not an official blind schedule beyond 10/20.
        bb = 20 * (2 ** (self.hands_played // 25))
        for i, p in enumerate(ps):
            p.update(bet=0, committed=0, folded=i not in alive, all_in=False)
        self.history = []
        deck = self.rng.sample(range(52), len(alive) * 2 + 5)
        self.holes = [None] * len(ps)
        pos = 0
        for i in sorted(alive):
            self.holes[i] = deck[pos:pos + 2]
            pos += 2
        runout = deck[pos:pos + 5]
        self.pay(sb, bb // 2)
        self.pay(big, bb)
        first = sb if len(alive) == 2 else self.next_live(big, alive)
        self.betting(first, [], "preflop", bb, bb)
        for n, street in ((3, "flop"), (4, "turn"), (5, "river")):
            if sum(not p["folded"] for p in ps) <= 1:
                break
            for p in ps:
                p["bet"] = 0
            first = self.next_live(self.button, {i for i, p in enumerate(ps) if not p["folded"]})
            self.betting(first, runout[:n], street, bb, 0)
        contenders = [i for i, p in enumerate(ps) if not p["folded"]]
        if len(contenders) == 1:
            ps[contenders[0]]["stack"] += sum(p["committed"] for p in ps)
        else:
            self.showdowns += 1
            award_pots(ps, runout, self.button, self.holes)
        eliminated = [i for i in alive if ps[i]["stack"] == 0]
        if eliminated:
            # Simultaneous eliminations are ordered by chips at start of hand;
            # equal initial stacks share their finishing places in local metrics.
            eliminated.sort(key=lambda i: before_stacks[i])
            groups = []
            for i in eliminated:
                if groups and before_stacks[groups[-1][0]] == before_stacks[i]:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            self.elimination_groups.extend(groups)
        self.hands_played += 1
        if sum(p["stack"] for p in ps) != self.total or any(p["stack"] < 0 for p in ps):
            raise AssertionError("chip conservation violation")
        return True

    def run(self):
        while self.hands_played < self.max_hands and self.hand():
            pass
        return {"stacks": [p["stack"] for p in self.players], "hands": self.hands_played,
                "elimination_groups": self.elimination_groups, "showdowns": self.showdowns,
                "actions": [dict(c) for c in self.actions], "timeouts": self.timeouts}
