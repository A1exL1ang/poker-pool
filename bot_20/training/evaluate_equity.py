"""Evaluate a frozen exported model on independently seeded unseen boards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pokerbot.equity import EquityEstimator
from pokerbot.features import extract_features
from training.fit_equity import grouped_metrics, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", default="pokerbot/assets/equity_model.json.gz")
    parser.add_argument("--preflop", default="pokerbot/assets/preflop.json")
    parser.add_argument("--report", default="measurements/equity_heldout.json")
    args = parser.parse_args()
    model = EquityEstimator(args.model, args.preflop)
    if not model.model_loaded or not model.preflop_loaded:
        raise RuntimeError("Both trained model and preflop table are required for final evaluation")
    test = np.load(args.test, allow_pickle=False)
    prediction = np.empty(len(test))
    times = []
    for i, row in enumerate(test):
        hole = (int(row[0]), int(row[1]))
        board = tuple(int(c) for c in row[2:2 + int(row[7])])
        n, fraction, pressure = int(row[8]), float(row[9]), float(row[10])
        begin = time.perf_counter_ns()
        prediction[i] = model.estimate(hole, board, n, fraction, pressure)
        times.append(time.perf_counter_ns() - begin)
    report = {"purpose": "Final held-out equity accuracy, independent boards and Monte Carlo labels",
        "model_sha256": sha256(args.model), "preflop_sha256": sha256(args.preflop), "test_sha256": sha256(args.test),
        "metrics": grouped_metrics(test, prediction),
        "latency_us": {"median": float(np.median(times) / 1000), "p95": float(np.quantile(times, .95) / 1000),
                       "p99": float(np.quantile(times, .99) / 1000), "max": float(max(times) / 1000),
                       "over_1ms": sum(t > 1000000 for t in times), "calls": len(times)}}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
