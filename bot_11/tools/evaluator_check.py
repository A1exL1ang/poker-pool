"""Reproducible evaluator checks and latency microbenchmark.

python -m tools.evaluator_check --hands 100000 --cross-check 10000
python -m tools.evaluator_check --exhaustive
"""

import argparse
import itertools
import json
import platform
import random
import time

from pokerbot.evaluator import CATEGORY_NAMES, evaluate, evaluate5


def run(args):
    rng = random.Random(args.seed)
    hands = [rng.sample(range(52), 7) for _ in range(args.hands)]
    for hand in hands[:1000]:
        evaluate(hand)
    start = time.perf_counter_ns()
    checksum = sum(evaluate(hand) for hand in hands)
    elapsed = time.perf_counter_ns() - start
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "seed": args.seed, "benchmark_hands": args.hands,
              "mean_evaluate7_us": elapsed / (1000 * args.hands), "checksum": checksum}
    for hand in hands[:args.cross_check]:
        reference = max(map(evaluate5, itertools.combinations(hand, 5)))
        if evaluate(hand) != reference:
            raise AssertionError((hand, evaluate(hand), reference))
    result["reference_cross_checked"] = min(args.cross_check, len(hands))
    if args.exhaustive:
        start = time.perf_counter()
        counts = [0] * 9
        for hand in itertools.combinations(range(52), 5):
            counts[evaluate(hand) >> 20] += 1
        expected = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
        if counts != expected:
            raise AssertionError((counts, expected))
        result["exhaustive_5card_category_counts"] = dict(zip(CATEGORY_NAMES, counts))
        result["exhaustive_seconds"] = time.perf_counter() - start
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11003)
    parser.add_argument("--hands", type=int, default=100000)
    parser.add_argument("--cross-check", type=int, default=10000)
    parser.add_argument("--exhaustive", action="store_true")
    run(parser.parse_args())
