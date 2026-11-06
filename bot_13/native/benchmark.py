"""Microbenchmark ctypes and native modeled equity, including Python overhead."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bot13.native import NativeEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = NativeEvaluator()
    result = {"platform": platform.platform(), "python": platform.python_version(), "iterations": args.iterations, "cases": {}}
    for board, street in (([1, 18, 39], "flop"), ([1, 18, 39, 44], "turn"), ([1, 18, 39, 44, 10], "river")):
        for opponents in (1, 3, 7):
            for samples in (64, 128):
                specs = [(0.20, .70, .08, .03, .25)]*opponents
                for seed in range(20):
                    engine.modeled_equity([48, 49], board, specs, seed=seed, samples=samples)
                times, accepted = [], []
                for seed in range(args.iterations):
                    start = time.perf_counter_ns()
                    data = engine.modeled_equity([48, 49], board, specs, seed=seed+1300, samples=samples)
                    times.append((time.perf_counter_ns()-start)/1000)
                    accepted.append(data["accepted_fraction"])
                times.sort()
                result["cases"][f"{street}_{opponents}opp_{samples}samples"] = {
                    "median_us": statistics.median(times), "p95_us": times[int(.95*len(times))],
                    "p99_us": times[int(.99*len(times))], "max_us": max(times),
                    "over_1ms": sum(t > 1000 for t in times),
                    "mean_accepted_fraction": statistics.mean(accepted),
                }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
