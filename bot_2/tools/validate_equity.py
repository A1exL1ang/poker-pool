"""Cross-language feature audit and independent-seed high-sample equity checks."""
import argparse
import json
import random
import statistics
import time
from pathlib import Path
from pokerbot.fast_equity import features, equity, METADATA
from pokerbot.equity import CUTOFFS, postflop_equity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=1200)
    parser.add_argument("--output", default="benchmarks/equity_validation.json")
    args = parser.parse_args()
    audits = 0
    for line in Path("assets/equity_audit.jsonl").read_text().splitlines():
        d = json.loads(line)
        actual = features(d["hole"], d["board"], d["range_index"], d["pressure"])
        assert tuple(d["features"]) == actual, (d, actual)
        audits += 1
    rng = random.Random(816271)
    errors = []
    per_street = {i: [] for i in (1, 2, 3)}
    worst = []
    start = time.perf_counter()
    for i in range(args.states):
        length = 3 + i % 3
        c = tuple(rng.sample(range(52), length + 2))
        ri = rng.randrange(10)
        pressure = rng.choice((0, .5, .85, .97))
        n = rng.choice((1, 2, 3, 5, 7))
        estimate, hu, _ = equity(c[:2], c[2:], CUTOFFS[ri], n, pressure)
        reference, ref_hu, _ = postflop_equity(c[:2], c[2:], CUTOFFS[ri], n, pressure,
                                               rng.randrange(1, 2**32), runouts=64, samples=32)
        error = estimate - reference
        errors.append(error)
        per_street[length - 2].append(error)
        worst.append({"hole": c[:2], "board": c[2:], "width": CUTOFFS[ri],
                      "pressure": pressure, "opponents": n,
                      "estimate": estimate, "reference": reference, "error": error})
    def metrics(v):
        return {"states": len(v), "rmse": (statistics.mean(e*e for e in v))**.5,
                "mae": statistics.mean(abs(e) for e in v), "bias": statistics.mean(v),
                "p95_absolute_error": sorted(abs(e) for e in v)[int(.95*len(v))]}
    report = {"model_metadata": METADATA, "cross_language_audits_passed": audits,
              "independent_seed": 816271, "reference_samples_per_state": 2048,
              "overall": metrics(errors),
              "by_street": {k: metrics(v) for k, v in per_street.items()},
              "worst_10": sorted(worst, key=lambda d: abs(d["error"]), reverse=True)[:10],
              "seconds": time.perf_counter() - start,
              "note": "Synthetic-state approximation error vs high-sample local conditional range model; neither is a solved poker strategy. Reference retains finite Monte Carlo error."}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "worst_10"}, indent=2))


if __name__ == "__main__":
    main()
