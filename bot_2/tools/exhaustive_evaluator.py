"""Full five-card histogram and distinct-score validation, ~2.6 million hands."""
import itertools
import json
import time
from collections import Counter
from pokerbot.cards import evaluate

start = time.perf_counter()
counts = Counter()
scores = set()
for hand in itertools.combinations(range(52), 5):
    score = evaluate(hand)
    counts[score >> 20] += 1
    scores.add(score)
expected = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
assert [counts[i] for i in range(9)] == expected, counts
assert len(scores) == 7462, len(scores)
print(json.dumps({"hands": sum(counts.values()), "category_histogram": dict(counts),
                  "distinct_scores": len(scores), "seconds": time.perf_counter() - start,
                  "passed": True}, indent=2))
