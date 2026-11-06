"""Standard-library latency check for the exact packaged runtime components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokerbot.equity import EquityEstimator
from pokerbot.features import extract_features


def summarize(times):
    values = sorted(times)
    return {"calls": len(values), "median_us": statistics.median(values) / 1000,
            "p95_us": values[min(len(values) - 1, int(len(values) * .95))] / 1000,
            "p99_us": values[min(len(values) - 1, int(len(values) * .99))] / 1000,
            "max_us": values[-1] / 1000, "over_1ms": sum(v > 1000000 for v in values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=90491)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--report", default="measurements/equity_runtime_latency.json")
    args = parser.parse_args()
    init_start = time.perf_counter_ns()
    model = EquityEstimator(args.model)
    init_ns = time.perf_counter_ns() - init_start
    if not args.allow_fallback and not (model.model_loaded and model.preflop_loaded):
        raise RuntimeError("Packaged trained assets are required; --allow-fallback is only for explicit smoke tests")
    rng = random.Random(args.seed)
    rows = []
    for _ in range(args.calls + 500):
        sample = rng.sample(range(52), rng.choice((5, 6, 7)))
        rows.append((tuple(sample[:2]), tuple(sample[2:]), rng.randint(1, 7), rng.uniform(.05, 1), rng.random()))
    for hole, board, n, fraction, pressure in rows[:500]:
        model.estimate(hole, board, n, fraction, pressure)
    single, double, preflop = [], [], []
    for hole, board, n, fraction, pressure in rows[500:]:
        start = time.perf_counter_ns()
        model.estimate(hole, board, n, fraction, pressure)
        single.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        features = extract_features(hole, board)
        model.predict(features, n, fraction, pressure)
        model.predict(features, n, max(.05, fraction * .65), min(1, pressure + .2))
        double.append(time.perf_counter_ns() - start)
        start = time.perf_counter_ns()
        model.estimate(hole, (), n, fraction, 0)
        preflop.append(time.perf_counter_ns() - start)
    report = {"purpose": "Equity components only; full Agent.act needs a separate benchmark",
              "platform": platform.platform(), "python": platform.python_version(), "seed": args.seed,
              "model_loaded": model.model_loaded, "preflop_loaded": model.preflop_loaded, "trees": len(model.trees),
              "model": args.model or "pokerbot/assets/equity_model.json.gz", "model_feature_count": model.feature_count,
              "initialization_us": init_ns / 1000, "warmup_calls_excluded": 500,
              "postflop_one_estimate": summarize(single), "postflop_features_plus_two_predictions": summarize(double),
              "preflop": summarize(preflop)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
