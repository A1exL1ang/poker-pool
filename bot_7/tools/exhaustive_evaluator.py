"""Exhaust all 2,598,960 five-card combinations and check category counts."""
import itertools
import json
from pathlib import Path
import time
from bot7.evaluator import evaluate


def main():
    start = time.perf_counter()
    counts = [0] * 9
    for cards in itertools.combinations(range(52), 5):
        counts[evaluate(cards) >> 20] += 1
    expected = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
    assert counts == expected, (counts, expected)
    report = {"hands": sum(counts), "counts_by_category": counts, "expected": expected,
              "elapsed_seconds": time.perf_counter() - start, "passed": True}
    Path("benchmarks/exhaustive_evaluator.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
