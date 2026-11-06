"""Exhaustive five-card category counts, plus every three/four-card board."""
from collections import Counter
import itertools
import json
from math import comb
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot.cards import evaluate


def main():
    start = time.perf_counter()
    expected = [0] * 9
    expected[8] = 4 * 10
    expected[7] = 13 * 48
    expected[6] = 13 * comb(4, 3) * 12 * comb(4, 2)
    expected[5] = 4 * comb(13, 5) - expected[8]
    expected[4] = 10 * 4**5 - expected[8]
    expected[3] = 13 * comb(4, 3) * comb(12, 2) * 4**2
    expected[2] = comb(13, 2) * comb(4, 2)**2 * 11 * 4
    expected[1] = 13 * comb(4, 2) * comb(12, 3) * 4**3
    expected[0] = comb(52, 5) - sum(expected)
    counts = Counter(evaluate(cs) >> 20 for cs in itertools.combinations(range(52), 5))
    assert [counts[i] for i in range(9)] == expected, (counts, expected)
    partial = {}
    for n in (3, 4):
        partial[n] = Counter(evaluate(cs) >> 20 for cs in itertools.combinations(range(52), n))
        assert sum(partial[n].values()) == comb(52, n)
    report = {'five_card_hands': comb(52, 5), 'categories': dict(counts), 'expected': expected,
              'all_three_and_four_card_boards': partial, 'passed': True,
              'seconds': time.perf_counter()-start}
    (ROOT / 'evidence/evaluator_census.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
