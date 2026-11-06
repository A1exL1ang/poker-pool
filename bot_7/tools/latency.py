"""Measure direct-call and dictionary-boundary latency, including stress states."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import random
import time
from bot7.agent import Agent
from bot7.cards import card
from bot7.interface import Event, Observation, Player
from .benchmark import summarize
from .baselines import Baseline
from .simulator import Table
from .evidence import fingerprint


def corpus():
    states = []
    for seed in range(8):
        table = Table([Baseline("random", seed * 8 + i) for i in range(8)], seed=seed, table_id=f"latency-{seed}")
        table.capture = True
        for h in range(15):
            table.play_hand(20)
        states.extend(obs for _, obs in table.observations)
    # Stress all three postflop streets with a full table and with nontrivial ICM.
    for board in ("2c 7d Jh", "2c 7d Jh Qs", "2c 7d Jh Qs 9d"):
        cards = tuple(map(card, board.split()))
        states.append(Observation(tuple(map(card, ("As", "Kh"))), cards,
                                  tuple(Player(i, 1600, 0, 400) for i in range(8)),
                                  0, 7, 3200, 0, 20, 1600, ("check", "raise"), table_id="stress-eight"))
        states.append(Observation(tuple(map(card, ("As", "Kh"))), cards,
                                  tuple(Player(i, 1000 if i != 1 else 0, 0 if i != 1 else 1000,
                                               0 if i != 1 else 1000, i >= 2) for i in range(8)),
                                  0, 7, 1000, 1000, 2000, 1000, ("fold", "call"),
                                  payouts=(50, 30, 20, 0, 0, 0, 0, 0), table_id="stress-icm"))
        states.append(Observation(tuple(map(card, ("As", "Kh"))), cards,
                                  tuple(Player(i, 1000 if i != 1 else 700, 0 if i != 1 else 300,
                                               0 if i != 1 else 300, i >= 2) for i in range(8)),
                                  0, 7, 300, 300, 600, 1000, ("fold", "call", "raise"),
                                  payouts=(80, 42, 23, 15, 9, 5, 2, 1), table_id="stress-full-icm"))
        amounts = [100, 800, 700, 600, 500, 400, 300, 200]
        states.append(Observation(tuple(map(card, ("As", "Kh"))), cards,
                                  tuple(Player(i, 700 if i == 0 else 0, amounts[i], amounts[i], False, i != 0) for i in range(8)),
                                  0, 7, sum(amounts), 700, 1500, 800, ("fold", "call"),
                                  table_id="stress-sidepots"))
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--comparisons", type=int, default=48)
    parser.add_argument("--output", default="benchmarks/latency.json")
    args = parser.parse_args()
    runtime_fingerprint = fingerprint()
    states = corpus()
    rng = random.Random(7)
    selected = [states[rng.randrange(len(states))] for _ in range(args.iterations)]
    agent = Agent(comparisons=args.comparisons)
    by_street = {s: [] for s in ("preflop", "flop", "turn", "river")}
    times = []
    for obs in selected[:500]:
        agent.act(obs)
    aggregate_cpu_start = time.thread_time_ns()
    for obs in selected:
        start = time.perf_counter_ns()
        action = agent.act(obs)
        elapsed = time.perf_counter_ns() - start
        assert action.action in obs.legal_actions
        assert agent.errors == 0
        times.append(elapsed)
        by_street[obs.street].append(elapsed)
    aggregate_cpu_ns = time.thread_time_ns() - aggregate_cpu_start
    dict_times = []
    for obs in selected[:2000]:
        value = asdict(obs)
        start = time.perf_counter_ns()
        agent.act_dict(value)
        assert agent.errors == 0
        dict_times.append(time.perf_counter_ns() - start)
    stress = []
    for obs in (state for state in states if state.table_id.startswith("stress-")):
        values = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            agent.act(obs)
            assert agent.errors == 0
            values.append(time.perf_counter_ns() - start)
        stress.append({"table": obs.table_id, "street": obs.street, "ns": summarize(values),
                       "over_1ms": sum(t > 1_000_000 for t in values)})
    report = {"platform": platform.platform(), "python": platform.python_version(), "comparisons": args.comparisons, "runtime_fingerprint": runtime_fingerprint,
              "corpus_states": len(states), "iterations": args.iterations, "direct_call_ns": summarize(times),
              "aggregate_thread_cpu_ns_per_call": aggregate_cpu_ns / args.iterations,
              "cpu_timing_note": "Thread CPU time is aggregated because Windows exposes coarse per-call accounting; this mean includes harness overhead.",
              "dict_boundary_ns": summarize(dict_times), "by_street_ns": {s: summarize(v) for s, v in by_street.items()},
              "over_1ms": sum(t > 1_000_000 for t in times), "agent_errors": agent.errors, "stress": stress,
              "timing_note": "Wall-clock perf_counter_ns, GC enabled, includes Python strategy only. OS scheduling can produce outliers; no portable hard-real-time guarantee."}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

