"""Reproducible latency evidence on recorded and adversarial observations."""
import argparse
import gc
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from bot6.agent import Agent
from bot6.interface import State
from tools.simulate import percentile


def synthetic_states():
    rng = random.Random(600600)
    result = []
    for n in (2, 4, 8):
        for size in (0, 3, 4, 5):
            for pressure in (0, 1, 2):
                for rep in range(20):
                    cards = rng.sample(range(52), 2 + size)
                    hero = rep % n
                    board = cards[2:]
                    street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[size]
                    players = [{"seat": i, "stack": 1000, "bet": 0 if i == hero else 100 if pressure else 0,
                                "contributed": 100 if i == hero else 200 if pressure else 100} for i in range(n)]
                    history = [{"seat": i, "street": "preflop", "action": "raise" if i == (hero + 1) % n else "call", "amount": 100} for i in range(n)]
                    if size:
                        for p in range(pressure):
                            history.extend({"seat": i, "street": street, "action": "raise", "amount": 100} for i in range(n) if i != hero)
                    o = {"table_id": 777, "hand_id": rep, "hero": hero, "hole_cards": cards[:2], "board": board,
                         "button": (hero + 1) % n, "big_blind": 20,
                         "pot": sum(p["contributed"] for p in players), "to_call": 100 if pressure else 0,
                         "min_raise_to": 200 if pressure else 20, "max_raise_to": 1000,
                         "legal_actions": ["fold", "call", "raise"] if pressure else ["check", "raise"],
                         "players": players, "history": history}
                    result.append(o)
    # ICM path: eight seated players, one all-in opponent, others folded.
    for size in (0, 3, 4, 5):
        for rep in range(30):
            cards = rng.sample(range(52), 2 + size)
            ps = [{"seat": 0, "stack": 1000, "contributed": 100, "bet": 0},
                  {"seat": 1, "stack": 0, "contributed": 1100, "bet": 1000, "all_in": True}]
            ps += [{"seat": i, "stack": 1000, "contributed": 100, "bet": 0, "folded": True} for i in range(2, 8)]
            result.append({"table_id": 888, "hand_id": rep, "hero": 0, "hole_cards": cards[:2], "board": cards[2:],
                           "button": 7, "big_blind": 20, "pot": 1800, "to_call": 1000, "min_raise_to": 2000,
                           "max_raise_to": 1000, "legal_actions": ["fold", "call"], "players": ps,
                           "payouts": [50, 30, 20, 0, 0, 0, 0, 0], "history": []})
    return result


def distribution(values):
    return {"count": len(values), "mean_us": statistics.mean(values) / 1000,
            "p50_us": percentile(values, .50) / 1000, "p95_us": percentile(values, .95) / 1000,
            "p99_us": percentile(values, .99) / 1000, "p999_us": percentile(values, .999) / 1000,
            "max_us": max(values) / 1000, "over_1ms": sum(x > 1_000_000 for x in values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--time-budget-us", type=int, default=500)
    parser.add_argument("--typed", action="store_true")
    parser.add_argument("--output", default="benchmarks/latency.json")
    parser.add_argument("--states", default="benchmarks/states.jsonl")
    args = parser.parse_args()
    synthetic = synthetic_states()
    states_path = Path(args.states)
    recorded = [json.loads(line) for line in states_path.read_text().splitlines()] if states_path.exists() else []
    rng = random.Random(6006)
    if len(recorded) > 4000:
        recorded = rng.sample(recorded, 4000)
    states = recorded + synthetic * 3
    rng.shuffle(states)
    if args.typed:
        states = [State.from_dict(s) for s in states]
    agent = Agent(6060, args.budget, args.time_budget_us)
    for o in states[:1000]:
        agent.act(o)
    times, effective_samples = [], []
    categories = {}
    slowest = []
    gc.collect()
    start = time.perf_counter()
    cpu_start = time.process_time()
    for i in range(args.iterations):
        o = states[i % len(states)]
        wall0 = time.perf_counter_ns()
        agent.act(o)
        elapsed = time.perf_counter_ns() - wall0
        times.append(elapsed)
        board = o.board if args.typed else o["board"]
        if board and agent.last_samples is not None:
            effective_samples.append(agent.last_samples)
        payout = bool(o.payouts) if args.typed else bool(o.get("payouts"))
        category = ("ICM-" if payout else "") + {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)]
        categories.setdefault(category, []).append(elapsed)
        if elapsed > 1_000_000:
            slowest.append({"iteration": i, "wall_us": elapsed / 1000, "category": category,
                            "players": len(o.players if args.typed else o["players"])})
    result = {"python": sys.version, "platform": platform.platform(), "budget": args.budget,
              "time_budget_us": args.time_budget_us,
              "mean_postflop_effective_samples": statistics.mean(effective_samples),
              "input": "validated State" if args.typed else "dictionary including card parsing and validation",
              "gc_enabled": gc.isenabled(), "iterations": args.iterations,
              "source_states": len(states), "recorded_states": len(recorded), "synthetic_states": len(synthetic),
              "wall_seconds": time.perf_counter() - start, "process_cpu_seconds": time.process_time() - cpu_start,
              "wall": distribution(times),
              "by_category": {k: distribution(v) for k, v in categories.items()},
              "slowest": sorted(slowest, key=lambda x: x["wall_us"], reverse=True)[:30],
              "scope": "Persistent process, warm imports/assets; no JSON I/O or process startup. Synthetic cases include heavy multiway pressure and 8-seat ICM. GC enabled. Wall-clock per-action timing; aggregate process CPU only because this Windows runtime quantizes thread CPU time at about 15.6 ms."}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "slowest"}, indent=2))


if __name__ == "__main__":
    main()
