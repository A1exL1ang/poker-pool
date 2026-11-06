"""Reproducible local no-limit Hold'em test environment, not an official kit.

Default blind progression and payouts are explicitly synthetic test settings.
All chip amounts are integers. Supports heads-up blinds, side pots, short
all-ins, cumulative reopening, split pots, and a hand-limit ICM allocation.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import time

from pokerbot.agent import Agent, opening_score
from pokerbot.cards import evaluate, hand_index
from pokerbot.equity import PERCENTILE, Equity
from pokerbot.icm import icm_equity


def award_pots(players, board, button=0):
    """Return awards, including uncalled excess, without mutating the input."""
    awards = [0] * len(players)
    levels = sorted(set(p["committed"] for p in players if p["committed"] > 0))
    previous = 0
    scores = {i: evaluate(p["hole"] + board) for i, p in enumerate(players) if not p["folded"]}
    for level in levels:
        contributors = [i for i, p in enumerate(players) if p["committed"] >= level]
        amount = (level - previous) * len(contributors)
        eligible = [i for i in contributors if not players[i]["folded"]]
        if not eligible:
            raise AssertionError("side pot has no eligible player")
        best = max(scores[i] for i in eligible)
        winners = sorted((i for i in eligible if scores[i] == best), key=lambda i: (i - button - 1) % len(players))
        split, extra = divmod(amount, len(winners))
        for j, i in enumerate(winners):
            awards[i] += split + (j < extra)
        previous = level
    assert sum(awards) == sum(p["committed"] for p in players)
    return awards


class Baseline:
    """Small independent test policies. They are deliberately not expert bots."""
    def __init__(self, style, seed):
        self.style = style
        self.rng = random.Random(seed)

    def act(self, obs):
        legal = obs["legal_actions"]
        hole, board = obs["hole_cards"], obs["board"]
        call = legal.get("call", 0)
        pot = obs["pot"]
        odds = call / max(1, pot + call)
        stack = obs["players"][obs["hero"]]["stack"]
        r = self.rng.random()
        strength = 1 - PERCENTILE[hand_index(*hole)]
        if board:
            cat = evaluate(hole + board) >> 20
            high = max(c >> 2 for c in board)
            top = any(c >> 2 == high for c in hole)
            over = hole[0] >> 2 == hole[1] >> 2 and hole[0] >> 2 > high
            strength = (0.18, 0.46, 0.72, 0.85, 0.90, 0.94, 0.98, 0.995, 1.0)[cat]
            if cat == 1 and (top or over):
                strength += 0.17
        if self.style == "random":
            raise_it, call_it = r < 0.10, r < 0.70
        elif self.style == "station":
            raise_it = strength > 0.94 and r < 0.40
            call_it = odds <= 0.34 or strength > 0.60 or call <= 2 * obs["big_blind"]
        elif self.style == "tight":
            raise_it = strength > (0.91 if not board else 0.78)
            call_it = strength > (0.80 if not board else 0.49 + 0.35 * odds)
        elif self.style == "pressure":
            raise_it = strength > (0.65 if not board else 0.58) or r < 0.20
            call_it = strength > (0.52 if not board else 0.30 + 0.50 * odds)
        elif self.style == "fold":
            raise_it = call_it = False
        else:
            raise ValueError("unknown baseline")
        if raise_it and "raise" in legal:
            bounds = legal["raise"]
            target = max(bounds["min"], round((2.5 * obs["big_blind"]) if not board else obs["players"][obs["hero"]]["bet"] + call + 0.65 * (pot + call)))
            return {"action": "raise", "amount": min(bounds["max"], target)}
        if legal.get("check"):
            return {"action": "check", "amount": 0}
        if call_it and call:
            return {"action": "call", "amount": call}
        return {"action": "fold", "amount": 0}


class Table:
    def __init__(self, bots, seed=1, stack=2000, hand_limit=200, blind_levels=None, level_hands=25, payouts=None, table_id=None, enforce_timeout=False, capture=None):
        self.bots = bots
        self.n = len(bots)
        self.rng = random.Random(seed)
        self.players = [{"stack": stack, "bet": 0, "committed": 0, "folded": False, "hole": []} for _ in bots]
        self.total_chips = stack * self.n
        self.hand_limit = hand_limit
        self.blind_levels = blind_levels or (20, 40, 80, 160, 320, 640, 1280, 2560)
        self.level_hands = level_hands
        self.payouts = tuple(payouts) if payouts is not None else None
        self.points = [0.0] * self.n
        self.button = seed % self.n
        self.hand_id = 0
        self.table_id = table_id if table_id is not None else f"local-{seed}"
        self.history = []
        self.board = []
        self.street = "preflop"
        self.latencies = [[] for _ in bots]
        self.actions = [Counter() for _ in bots]
        self.errors = []
        self.enforce_timeout = enforce_timeout
        self.capture = capture
        self.current = 0
        self.last_full_raise = 0
        self.acted_to = {}
        self.bb = 20
        self.hands_played = 0

    def alive(self):
        return [i for i, p in enumerate(self.players) if p["stack"] > 0]

    def next_seat(self, seat, seats):
        return min(seats, key=lambda i: (i - seat - 1) % self.n)

    def pay(self, i, amount):
        p = self.players[i]
        amount = min(amount, p["stack"])
        p["stack"] -= amount
        p["bet"] += amount
        p["committed"] += amount

    def check_chips(self):
        assert all(p["stack"] >= 0 and p["bet"] >= 0 and p["committed"] >= p["bet"] for p in self.players)
        assert sum(p["stack"] + p["committed"] for p in self.players) == self.total_chips

    def legal_actions(self, i):
        p = self.players[i]
        need = max(0, self.current - p["bet"])
        legal = {"fold": True, "call": min(need, p["stack"])} if need else {"check": True}
        maximum = p["bet"] + p["stack"]
        reopened = i not in self.acted_to or self.current - self.acted_to[i] >= self.last_full_raise
        other_can_bet = any(j != i and not q["folded"] and q["stack"] > 0 for j, q in enumerate(self.players))
        if maximum > self.current and reopened and other_can_bet:
            legal["raise"] = {"min": min(maximum, self.current + self.last_full_raise), "max": maximum}
        return legal

    def observe(self, i, legal):
        result = {
            "table_id": self.table_id, "hand_id": self.hand_id, "hero": i,
            "button": self.button, "street": self.street,
            "hole_cards": list(self.players[i]["hole"]), "board": list(self.board),
            "big_blind": self.bb, "pot": sum(p["committed"] for p in self.players),
            "current_bet": self.current,
            "players": [{k: p[k] for k in ("stack", "bet", "committed", "folded")} for p in self.players],
            "legal_actions": legal, "history": list(self.history),
        }
        if self.payouts is not None:
            # Payouts for currently remaining places, first through last.
            n = sum(p["stack"] + p["committed"] > 0 for p in self.players)
            result["payouts"] = list(self.payouts[:n])
        return result

    def betting_round(self, first):
        self.current = max(p["bet"] for p in self.players)
        if self.street == "preflop" and sum(not p["folded"] and p["stack"] > 0 for p in self.players) > 1:
            self.current = max(self.current, self.bb)
        self.last_full_raise = self.bb
        self.acted_to = {}
        pending = {i for i, p in enumerate(self.players) if not p["folded"] and p["stack"] > 0}
        cursor = (first - 1) % self.n
        count = 0
        while pending:
            active = [i for i, p in enumerate(self.players) if not p["folded"]]
            if len(active) <= 1:
                return
            actors = [i for i in active if self.players[i]["stack"] > 0]
            if len(actors) == 1 and self.players[actors[0]]["bet"] >= self.current:
                return
            i = self.next_seat(cursor, pending)
            cursor = i
            p = self.players[i]
            legal = self.legal_actions(i)
            obs = self.observe(i, legal)
            t = time.perf_counter_ns()
            action = self.bots[i].act(obs)
            elapsed = time.perf_counter_ns() - t
            self.latencies[i].append(elapsed / 1000)
            if self.capture is not None:
                self.capture(obs, action, elapsed / 1000)
            if self.enforce_timeout and elapsed > 1_000_000:
                action = {"action": "check" if legal.get("check") else "fold", "amount": 0}
            kind = action.get("action")
            amount = action.get("amount", 0)
            if kind not in legal or kind in ("check", "fold") and not legal[kind]:
                raise AssertionError(f"illegal action {action}; legal {legal}")
            if kind == "call" and amount != legal["call"]:
                raise AssertionError("call amount must be incremental")
            if kind == "raise" and not legal["raise"]["min"] <= amount <= legal["raise"]["max"]:
                raise AssertionError("raise outside allowed bounds")
            if getattr(self.bots[i], "last_error", None):
                self.errors.append(self.bots[i].last_error)
            self.actions[i][kind] += 1
            pending.discard(i)
            if kind == "fold":
                p["folded"] = True
            elif kind == "call":
                self.pay(i, amount)
            elif kind == "raise":
                increment = amount - self.current
                self.pay(i, amount - p["bet"])
                if increment >= self.last_full_raise:
                    self.last_full_raise = increment
                self.current = amount
                pending = {j for j, q in enumerate(self.players) if j != i and not q["folded"] and q["stack"] > 0 and q["bet"] < self.current}
            self.acted_to[i] = self.current
            self.history.append({"player": i, "street": self.street, "action": kind, "amount": amount})
            self.check_chips()
            count += 1
            if count > 1000:
                raise AssertionError("betting round did not terminate")

    def play_hand(self):
        alive = self.alive()
        starts = [p["stack"] for p in self.players]
        self.button = self.next_seat(self.button, alive)
        self.bb = self.blind_levels[min(self.hand_id // self.level_hands, len(self.blind_levels) - 1)]
        self.board = []
        self.history = []
        self.street = "preflop"
        deck = list(range(52))
        self.rng.shuffle(deck)
        for i, p in enumerate(self.players):
            p.update(bet=0, committed=0, folded=i not in alive, hole=[deck.pop(), deck.pop()] if i in alive else [])
        if len(alive) == 2:
            sb = self.button
            bb_seat = self.next_seat(sb, alive)
            first = sb
        else:
            sb = self.next_seat(self.button, alive)
            bb_seat = self.next_seat(sb, alive)
            first = self.next_seat(bb_seat, alive)
        self.pay(sb, max(1, self.bb // 2))
        self.pay(bb_seat, self.bb)
        self.betting_round(first)
        for street, cards in (("flop", 3), ("turn", 1), ("river", 1)):
            if sum(not p["folded"] for p in self.players) == 1:
                break
            for p in self.players:
                p["bet"] = 0
            self.street = street
            deck.pop()  # burn
            self.board.extend(deck.pop() for _ in range(cards))
            self.betting_round(self.next_seat(self.button, [i for i, p in enumerate(self.players) if not p["folded"]]))
        active = [i for i, p in enumerate(self.players) if not p["folded"]]
        if len(active) == 1:
            awards = [0] * self.n
            awards[active[0]] = sum(p["committed"] for p in self.players)
        else:
            assert len(self.board) == 5
            awards = award_pots(self.players, self.board, self.button)
        for i, p in enumerate(self.players):
            p["stack"] += awards[i]
            p["committed"] = p["bet"] = 0
        self.check_chips()
        busted = sorted((i for i in alive if self.players[i]["stack"] == 0), key=lambda i: starts[i])
        if self.payouts:
            # Equal starting stacks share the tied finishing positions.
            position = len(alive) - 1
            while busted:
                same = [i for i in busted if starts[i] == starts[busted[0]]]
                payment = sum(self.payouts[position - j] if position - j < len(self.payouts) else 0 for j in range(len(same))) / len(same)
                for i in same:
                    self.points[i] = payment
                    busted.remove(i)
                position -= len(same)
        self.hand_id += 1
        self.hands_played += 1

    def run(self):
        while len(self.alive()) > 1 and self.hand_id < self.hand_limit:
            self.play_hand()
        alive = self.alive()
        if self.payouts:
            stacks = tuple(self.players[i]["stack"] for i in alive)
            payouts = self.payouts[:len(alive)]
            for j, i in enumerate(alive):
                self.points[i] = icm_equity(stacks, j, payouts)
        else:
            self.points = [p["stack"] / self.total_chips for p in self.players]
        return self.points


def summarize(values):
    values = sorted(values)
    n = len(values)
    return {"n": n, "mean": statistics.mean(values) if n else 0, "p50": values[n // 2] if n else 0, "p95": values[min(n - 1, int(n * .95))] if n else 0, "p99": values[min(n - 1, int(n * .99))] if n else 0, "max": max(values, default=0), "over_1000_us": sum(x > 1000 for x in values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--lineup", choices=("mixed", "tight", "station", "random", "pressure", "self"), default="mixed")
    parser.add_argument("--payouts", default="0.5,0.3,0.2,0,0,0,0,0", help="SYNTHETIC test payouts; comma separated, or 'chips'")
    parser.add_argument("--blinds", default="20,40,80,160,320,640,1280,2560", help="SYNTHETIC test big-blind progression")
    parser.add_argument("--enforce-timeout", action="store_true")
    parser.add_argument("--output", default="evidence/simulation.json")
    args = parser.parse_args()
    payouts = None if args.payouts == "chips" else tuple(float(x) for x in args.payouts.split(","))
    blinds = tuple(int(x) for x in args.blinds.split(","))
    points, chips, latency, errors, actions, tables = [], [], [], [], Counter(), []
    start = time.perf_counter()
    # Immutable table arrays can be shared; cache state belongs to each Agent.
    source = Equity()
    def new_equity():
        e = Equity(path="assets/not-a-real-asset")
        e.keys, e.values = source.keys, source.values
        return e
    for t in range(args.tables):
        seed = args.seed + t
        hero = t % 8
        styles = ["tight", "station", "pressure", "random", "tight", "station", "pressure"] if args.lineup == "mixed" else [args.lineup] * 7
        bots = [Baseline(style, seed * 17 + i) if style != "self" else Agent(seed * 17 + i, equity=new_equity()) for i, style in enumerate(styles)]
        bots.insert(hero, Agent(seed=seed, equity=new_equity()))
        table = Table(bots, seed=seed, hand_limit=args.hands, blind_levels=blinds, payouts=payouts, enforce_timeout=args.enforce_timeout)
        table.run()
        points.append(table.points[hero])
        chips.append(table.players[hero]["stack"])
        latency.extend(table.latencies[hero])
        errors.extend(table.errors)
        actions.update(table.actions[hero])
        tables.append({"seed": seed, "hero_seat": hero, "hands": table.hands_played, "points": table.points[hero], "chips": table.players[hero]["stack"]})
        if (t + 1) % 8 == 0:
            print(f"{t + 1}/{args.tables}: mean points {statistics.mean(points):.4f}; elapsed {time.perf_counter() - start:.1f}s", flush=True)
    stderr = statistics.stdev(points) / math.sqrt(len(points)) if len(points) > 1 else 0
    result = {"settings": vars(args), "note": "Local synthetic opponents, payout vector and later blind levels. Not official results or an estimate of contest win probability.", "mean_points": statistics.mean(points), "standard_error_points": stderr, "normal_95pct_interval": [statistics.mean(points) - 1.96 * stderr, statistics.mean(points) + 1.96 * stderr], "equal_share_points": (sum(payouts) if payouts else 1) / 8, "mean_final_chips": statistics.mean(chips), "latency_us": summarize(latency), "policy_errors": errors, "actions": dict(actions), "elapsed_seconds": time.perf_counter() - start, "tables": tables}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("tables", "policy_errors")}, indent=2))
    if errors:
        raise SystemExit(f"{len(errors)} policy errors")


if __name__ == "__main__":
    main()
