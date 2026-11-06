"""Held-out, continuous-range validation of the fixed preflop table."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokerbot.equity import EquityEstimator
from pokerbot.evaluator import card_text
from training.diagnose_equity import rows


def metrics(items):
    errors = sorted(abs(item["error"]) for item in items)
    mse = sum(item["error"] ** 2 for item in items) / len(items)
    noise = sum(item["stderr"] ** 2 for item in items) / len(items)
    return {"n": len(items), "mae": sum(errors) / len(errors), "rmse": math.sqrt(mse),
            "bias": sum(item["error"] for item in items) / len(items), "p95_abs_error": errors[int(len(errors) * .95)],
            "max_abs_error": errors[-1], "monte_carlo_noise_rmse": math.sqrt(noise),
            "noise_corrected_rmse_estimate": math.sqrt(max(0, mse - noise))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--report", default="measurements/preflop_heldout.json")
    args = parser.parse_args()
    model = EquityEstimator()
    if not model.preflop_loaded:
        raise RuntimeError("Trained preflop table required")
    results = []
    for row in rows(args.test):
        if row[7] != 0:
            raise ValueError("Expected preflop-only labels")
        hole = tuple(int(c) for c in row[:2])
        prediction = model.estimate(hole, (), int(row[8]), row[9], 0)
        results.append({"hole": " ".join(map(card_text, hole)), "opponents": int(row[8]), "range_fraction": row[9],
                        "target": row[11], "stderr": row[12], "prediction": prediction, "error": prediction - row[11]})
    report = {"purpose": "Independent Monte Carlo evaluation of fixed preflop table at continuous range fractions",
              "test_sha256": hashlib.sha256(Path(args.test).read_bytes()).hexdigest(), "all": metrics(results),
              "by_opponents": {str(n): metrics([r for r in results if r["opponents"] == n]) for n in range(1, 8)},
              "worst": sorted(results, key=lambda r: abs(r["error"]), reverse=True)[:20]}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": args.report, "metrics": report["all"]}))


if __name__ == "__main__":
    main()
