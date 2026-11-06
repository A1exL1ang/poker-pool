"""Measure real local Agent calls; report CPU and wall-clock tails separately.

An overrun is reported even if caused by scheduling. No claim is made about
the unavailable contest hardware, loader, or exact timing harness.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import platform
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter_ns, process_time_ns, get_clock_info
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokerbot.game import run_tournament, Table
from pokerbot.opponents import Opponent
from pokerbot.strategy import Agent
from tools.benchmark import BLINDS, PAYOUTS


def capture(seed, tables, per_street):
    groups = defaultdict(list)
    for k in range(tables):
        agents = [Agent(seed=seed + k)] + [Opponent(style, seed + k * 31 + i) for i, style in enumerate(("tag", "nit", "lag", "station", "maniac", "rollout", "balanced"))]
        r = run_tournament(agents, payouts=PAYOUTS["top3" if k % 2 else "linear"], blind_schedule=BLINDS["moderate"], seed=seed + k, capture_observations=4000)
        for o in r.observations:
            groups[o["street"]].append(o)
    rng = random.Random(seed)
    corpus = []
    for street, rows in groups.items():
        corpus.extend(rng.sample(rows, min(per_street, len(rows))))
    # Reproducible worst-case high-payout-count ICM/open-shove observations.
    for stacks in ((120,) * 8, (120, 125, 130, 135, 140, 145, 150, 155), (200,) * 8, (200, 210, 230, 250, 270, 310, 350, 390)):
        t = Table(stacks, payouts=PAYOUTS["linear"], capture_observations=16, seed=seed)
        t.play_hand([lambda o: {"action": "check" if not o["to_call"] else "fold"}] * 8)
        corpus.extend(t.observations)
    rng.shuffle(corpus)
    return corpus


def stats(ns):
    x = sorted(ns)
    def pct(p):
        return x[min(len(x) - 1, int(p * (len(x) - 1)))] / 1000
    return dict(calls=len(x), mean_us=statistics.mean(x) / 1000, median_us=pct(.5), p95_us=pct(.95), p99_us=pct(.99), p999_us=pct(.999), max_us=max(x) / 1000, over_1ms=sum(v > 1_000_000 for v in x), over_1ms_fraction=sum(v > 1_000_000 for v in x) / len(x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="measurements/latency_corpus.jsonl.gz")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--tables", type=int, default=20)
    p.add_argument("--per-street", type=int, default=800)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=141414)
    p.add_argument("--config", default="{}")
    p.add_argument("--output", default="measurements/latency.json")
    args = p.parse_args()
    path = Path(args.corpus)
    if args.generate or not path.exists():
        corpus = capture(args.seed, args.tables, args.per_street)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for o in corpus:
                f.write(json.dumps(o, separators=(",", ":")) + "\n")
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            corpus = [json.loads(line) for line in f]
    config = json.loads(args.config)
    agent = Agent(seed=args.seed, config=config)
    for o in corpus[:200]:
        agent.decide(o)
    wall, cpu = defaultdict(list), defaultdict(list)
    fallbacks = native_calls = native_fallbacks = 0
    for repeat in range(args.repeat):
        for o in corpus:
            # Reset outside the timed call: each observation pays for fresh
            # feature/range work, without artificial backwards hand-id resets.
            agent.reset()
            c0, w0 = process_time_ns(), perf_counter_ns()
            action = agent.decide(o)
            w, c = perf_counter_ns() - w0, process_time_ns() - c0
            wall[o["street"]].append(w)
            cpu[o["street"]].append(c)
            fallbacks += agent.diagnostics["budget_fallbacks"]
            native_calls += agent.diagnostics["equity_native_calls"]
            native_fallbacks += agent.diagnostics["equity_native_fallbacks"]
            if action["action"] not in o["legal_actions"]:
                raise AssertionError((action, o["legal_actions"]))
    cpu_usable = statistics.median(sum(cpu.values(), [])) > 0
    report = dict(platform=platform.platform(), python=sys.version, processor=platform.processor(), config=config, corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), corpus_size=len(corpus), repeat=args.repeat, wall=stats(sum(wall.values(), [])), cpu=stats(sum(cpu.values(), [])), cpu_quantiles_usable=cpu_usable, cpu_timer_resolution=get_clock_info("process_time").resolution, wall_by_street={k:stats(v) for k,v in wall.items()}, cpu_by_street={k:stats(v) for k,v in cpu.items()}, budget_fallbacks=fallbacks, native_calls=native_calls, native_fallbacks=native_fallbacks, assets=agent.diagnostics, notes="Import/constructor/corpus creation excluded. Fresh state per timed observation. Mixed real simulated observations plus8player full-rank ICM stress. Every wall overrun counts, including OS scheduling. Windows process CPU timer may be too coarse for per-call quantiles; see cpu_quantiles_usable.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
