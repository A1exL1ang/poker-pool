"""Validate generated inference against the reference on hands and branches."""
import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import equity_model as equity
from poker_eval import features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--output", default="measurements/equity_codegen_validation.json")
    args = parser.parse_args()
    if equity.asset_status()["inference_backend"] != "generated":
        raise RuntimeError(equity.asset_status())
    rng = random.Random(190602)
    hands = []
    synthetic = []
    for i in range(args.cases):
        n = 3 + i % 3
        cards = rng.sample(range(52), n+2)
        w = rng.choice((.05,.1,.2,.35,.6,1.0)) if i%2 else rng.uniform(.05,1)
        a = rng.choice((0,.35,.7,1.0)) if i%3 else rng.random()
        hands.append(features(cards[:2], cards[2:], w, a))
        f = [rng.randrange(16) for _ in range(43)]
        f[0], f[41], f[42] = n, w, a
        synthetic.append(f)
    maximum = 0.
    for f in hands + synthetic:
        fast = equity.predict_from_features(f)
        reference = equity._predict_reference(f)
        maximum = max(maximum, abs(fast-reference))
    if maximum > 1e-12:
        raise AssertionError(maximum)
    thresholds = set()
    for model in equity._MODEL["streets"].values():
        for tree in model["trees"]:
            for e in tree:
                if e[0] in (41,42):
                    thresholds.add((e[0],e[1]))
    boundary_cases = 0
    for index, value in sorted(thresholds):
        for x in (math.nextafter(value,-math.inf),value,math.nextafter(value,math.inf)):
            f = list(hands[boundary_cases % len(hands)])
            f[index] = x
            if equity.predict_from_features(f) != equity._predict_reference(f):
                raise AssertionError((index,x))
            boundary_cases += 1
    # Warm up first. Latencies include inference only, excluding feature extraction.
    for f in hands[:300]:
        equity.predict_from_features(f)
    times = []
    t0 = time.perf_counter_ns()
    for f in hands:
        start = time.perf_counter_ns()
        equity.predict_from_features(f)
        times.append(time.perf_counter_ns()-start)
    elapsed = time.perf_counter_ns()-t0
    times.sort()
    report = {"status":equity.asset_status(), "seed":190602, "actual_hands":len(hands),
              "random_feature_vectors":len(synthetic), "float32_boundary_cases":boundary_cases,
              "max_fast_reference_difference":maximum,
              "inference_latency_us":{"mean":elapsed/len(hands)/1000,
                  "median":statistics.median(times)/1000, "p99":times[int(.99*(len(times)-1))]/1000,
                  "max":max(times)/1000, "over_1ms":sum(t>1000000 for t in times)}}
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
