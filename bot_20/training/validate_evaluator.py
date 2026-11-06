"""Exhaustively verify all 2,598,960 five-card category counts.

Separate from routine unit tests because this deliberately enumerates the
entire five-card space. Counts follow elementary poker-hand combinatorics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokerbot.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="measurements/evaluator_exhaustive.json")
    args = parser.parse_args()
    expected = [
        (math.comb(13, 5) - 10) * (4 ** 5 - 4),
        13 * math.comb(4, 2) * math.comb(12, 3) * 4 ** 3,
        math.comb(13, 2) * math.comb(4, 2) ** 2 * 11 * 4,
        13 * math.comb(4, 3) * math.comb(12, 2) * 4 ** 2,
        10 * (4 ** 5 - 4),
        4 * (math.comb(13, 5) - 10),
        13 * math.comb(4, 3) * 12 * math.comb(4, 2),
        13 * 12 * 4,
        10 * 4,
    ]
    observed = [0] * 9
    start = time.perf_counter()
    for cards in itertools.combinations(range(52), 5):
        observed[evaluate(cards) >> 20] += 1
    report = {"hands": sum(observed), "category_counts": observed, "expected_counts": expected,
              "passed": observed == expected, "elapsed_seconds": time.perf_counter() - start}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if observed != expected:
        raise AssertionError("Five-card category counts do not match combinatorics")


if __name__ == "__main__":
    main()
