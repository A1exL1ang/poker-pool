"""Exhaustive standard 5-card category census; no external dependency."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import time
from bot6.cards import evaluate_mask, CARD_MASK


def main():
    start = time.perf_counter()
    counts = Counter(evaluate_mask(a | b | c | d | e) >> 24 for a, b, c, d, e in combinations(CARD_MASK, 5))
    expected = {0: 1302540, 1: 1098240, 2: 123552, 3: 54912, 4: 10200, 5: 5108, 6: 3744, 7: 624, 8: 40}
    assert counts == expected, (counts, expected)
    result = {"hands": sum(counts.values()), "counts": dict(sorted(counts.items())), "seconds": time.perf_counter() - start, "passed": True}
    Path("benchmarks/evaluator.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
