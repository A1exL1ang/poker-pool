"""Reproducible local benchmark CLI; no official interface/payout claims.

Example: python tournament.py --tables 100 --seed 900000 --label heldout \
             --lineup mixed --payouts top3 --output measurements/heldout.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time

from baselines import BASELINES
from simulator import Table, DEFAULT_LEVELS


EXPERIMENTAL_PAYOUTS = {
    "chip": None,
    "winner": [1, 0, 0, 0, 0, 0, 0, 0],
    "top3": [0.5, 0.3, 0.2, 0, 0, 0, 0, 0],
    "linear": [8, 7, 6, 5, 4, 3, 2, 1],
}
LEVELS = {
    "default": DEFAULT_LEVELS,
    "fast": ((10, 20), (20, 40), (40, 80), (80, 160), (160, 320), (320, 640), (640, 1280), (1280, 2560)),
    "slow": ((10, 20), (12, 25), (15, 30), (20, 40), (25, 50), (35, 70), (50, 100), (75, 150)),
}


def summarize(values):
    n = len(values)
    mean = statistics.fmean(values) if values else 0.0
    sd = statistics.stdev(values) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {"n": n, "mean": mean, "standard_error": se,
            "normal_95_interval": [mean - 1.96 * se, mean + 1.96 * se]}


def latency_summary(values, threshold=1_000_000):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def percentile(p):
        return values[min(len(values) - 1, int((len(values) - 1) * p))] / 1000
    return {"n": len(values), "median_us": percentile(0.5), "p95_us": percentile(0.95),
            "p99_us": percentile(0.99), "p999_us": percentile(0.999),
            "max_us": values[-1] / 1000, "over_1ms": sum(x > threshold for x in values)}


def run_benchmark(*, tables=100, seed=900000, lineup="mixed", payouts=None,
                  levels=DEFAULT_LEVELS, max_hands=200, hero_factory=None,
                  opponent_factory=None, enforce_time=False, label="unspecified", progress=False):
    if hero_factory is None:
        from agent import Agent
        hero_factory = Agent
    if lineup == "selfplay" and opponent_factory is None:
        from agent import Agent
        opponent_factory = Agent
    if lineup not in (*BASELINES, "mixed", "selfplay"):
        raise ValueError("unknown lineup")
    results, times = [], []
    sources = {}
    for filename in ("agent.py", "equity.py", "risk.py", "simulator.py", "baselines.py", "tournament.py"):
        path = Path(__file__).resolve().parent / filename
        if path.exists():
            sources[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    begin = time.perf_counter()
    for t in range(tables):
        table_seed = seed + t
        rng = random.Random(table_seed ^ 0xB017)
        hero_seat = rng.randrange(8)
        opponents = ["tight", "calling", "pressure", "random", "tight", "pressure", "shove"]
        if lineup == "mixed":
            rng.shuffle(opponents)
        else:
            opponents = [lineup] * 7
        agents, kinds = [], []
        j = 0
        for seat in range(8):
            instance_seed = table_seed * 19 + seat
            if seat == hero_seat:
                agents.append(hero_factory(seed=instance_seed))
                kinds.append("hero")
            else:
                kind = opponents[j]
                j += 1
                agents.append(opponent_factory(seed=instance_seed) if kind == "selfplay" else BASELINES[kind](seed=instance_seed))
                kinds.append(kind)
        table = Table(agents, seed=table_seed, levels=levels, max_hands=max_hands,
                      payouts=payouts, time_limit_ns=1_000_000 if enforce_time else None)
        result = table.run()
        result["hero_seat"] = hero_seat
        result["lineup"] = kinds
        result["hero_score"] = result["scores"][hero_seat]
        result["hero_chip_share"] = result["stacks"][hero_seat] / 16000
        result["hero_latency"] = latency_summary(table.latencies[hero_seat])
        times.extend(table.latencies[hero_seat])
        results.append(result)
        if progress and (t + 1) % 10 == 0:
            print(f"{t + 1}/{tables} tables, hero mean={statistics.fmean(x['hero_score'] for x in results):.5f}", flush=True)
    return {"label": label, "seed_start": seed, "tables": tables, "lineup": lineup,
            "payouts": payouts, "payout_warning": "Experimental vector, not official contest payouts. Null means terminal chip share.",
            "blind_levels": list(levels), "hands_per_level": 25, "max_hands": max_hands,
            "timing_enforced": enforce_time, "elapsed_seconds": time.perf_counter() - begin,
            "python": platform.python_version(), "platform": platform.platform(),
            "source_sha256_at_start": sources,
            "process_id": os.getpid(),
            "hero_score": summarize([r["hero_score"] for r in results]),
            "hero_chip_share": summarize([r["hero_chip_share"] for r in results]),
            "hero_latency": latency_summary(times),
            "hero_errors": sum(r["errors"][r["hero_seat"]] for r in results),
            "hero_invalid_actions": sum(r["invalid_actions"][r["hero_seat"]] for r in results),
            "hero_internal_fallbacks": sum(r["internal_fallbacks"][r["hero_seat"]] for r in results),
            "hero_observed_over_1ms": sum(r["observed_over_1ms"][r["hero_seat"]] for r in results),
            "hero_fallback_overruns": sum(r["overruns"][r["hero_seat"]] for r in results),
            "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, default=100)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--lineup", choices=[*BASELINES, "mixed", "selfplay"], default="mixed")
    parser.add_argument("--payouts", choices=EXPERIMENTAL_PAYOUTS, default="chip")
    parser.add_argument("--levels", choices=LEVELS, default="default")
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument("--enforce-time", action="store_true")
    parser.add_argument("--agent-module", default="agent")
    parser.add_argument("--hero-config", default="{}", help="JSON configuration dictionary passed as Agent(config=...)")
    parser.add_argument("--output", default="measurements/benchmark.json")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.hero_config)
    agent_class = importlib.import_module(args.agent_module).Agent
    def factory(seed):
        return agent_class(seed=seed, config=config)
    result = run_benchmark(tables=args.tables, seed=args.seed, lineup=args.lineup,
                           payouts=EXPERIMENTAL_PAYOUTS[args.payouts], levels=LEVELS[args.levels],
                           max_hands=args.max_hands, hero_factory=factory,
                           enforce_time=args.enforce_time, label=args.label, progress=args.progress)
    result["agent_module"] = args.agent_module
    result["hero_config"] = config
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
