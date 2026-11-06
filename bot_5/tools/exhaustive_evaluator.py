"""Exhaustive five-card category census against combinatorial counts."""
from itertools import combinations
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pokerbot.cards import evaluate


def main():
    expected = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
    counts = [0] * 9
    start = time.perf_counter()
    for hand in combinations(range(52), 5):
        counts[evaluate(hand) >> 20] += 1
    result = {"hands": sum(counts), "category_counts": counts, "expected_counts": expected, "passed": counts == expected, "seconds": time.perf_counter() - start}
    (ROOT / "evidence" / "exhaustive_evaluator.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if counts != expected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
