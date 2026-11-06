"""Measure import, typed decisions and JSON-dict adapter, wall and CPU time."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
tick = time.perf_counter_ns()
from agent import Agent, State, Player, Event
IMPORT_NS = time.perf_counter_ns() - tick


def examples(n=4000):
    rng = random.Random(4011)
    result = []
    for i in range(n):
        live = 2 + i % 7
        hero = i % live
        cs = rng.sample(range(52), 7)
        length = (0, 3, 4, 5)[i % 4]
        hole, board = tuple(cs[:2]), tuple(cs[2:2+length])
        current = rng.choice((0, 20, 50, 120, 400, 1000))
        own = 0 if current == 0 else rng.choice((0, current // 2, current))
        ps = tuple(Player(j, rng.randint(20, 5000), own if j == hero else current,
                          (own if j == hero else current) + 40, False, False) for j in range(live))
        h = ps[hero]
        events = tuple(Event(j, "preflop", "raise" if j == 0 else "call", 50, 20, 30+50*j) for j in range(live)) if i % 3 else ()
        result.append(State("benchmark", i, hero, i//live % live, hole, board, ps,
                            sum(p.contributed for p in ps), 20, current-own,
                            current+max(20,current-own), h.stack+h.committed, bool(i%5), events,
                            (.5,.3,.2) if i%2 else ()))
    return result


def summary(values):
    vs = sorted(values)
    return {"median_us": statistics.median(vs)/1000, "p95_us": vs[int(.95*len(vs))]/1000,
            "p99_us": vs[int(.99*len(vs))]/1000, "p999_us": vs[min(len(vs)-1,int(.999*len(vs)))]/1000,
            "max_us": max(vs)/1000, "over_1ms": sum(v>1000000 for v in vs)}


def run(states, rounds, adapter):
    bot = Agent()
    inputs = [asdict(s) for s in states] if adapter else states
    fn = bot.act if adapter else bot.decide
    # Warm interpreter and allocated structures, not the board cache.
    for s in inputs[:200]:
        fn(s)
    wall = []
    cpu_start = time.process_time_ns()
    total_start = time.perf_counter_ns()
    for _ in range(rounds):
        for s in inputs:
            t = time.perf_counter_ns()
            fn(s)
            wall.append(time.perf_counter_ns()-t)
    # Windows process_time_ns has ~15.625 ms granularity here. Per-decision
    # readings would be misleading; report an aggregate CPU average instead.
    cpu_total = time.process_time_ns() - cpu_start
    wall_total = time.perf_counter_ns() - total_start
    return {"decisions": len(wall), "wall": summary(wall),
            "aggregate_cpu_average_us": cpu_total / len(wall) / 1000,
            "aggregate_wall_average_us": wall_total / len(wall) / 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--output", default="evidence/latency.json")
    args = ap.parse_args()
    states = examples()
    report = {"python": sys.version, "platform": platform.platform(), "import_ms": IMPORT_NS/1e6,
              "gc_enabled": gc.isenabled(), "states": len(states),
              "typed": run(states, args.rounds, False), "dict_adapter": run(states, args.rounds, True)}
    captured = ROOT / "evidence/observations.jsonl"
    if captured.exists():
        real = [State.from_dict(json.loads(line)) for line in captured.read_text().splitlines()]
        report["simulator_observations"] = run(real, args.rounds, True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
