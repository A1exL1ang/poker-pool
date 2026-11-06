"""Optional exhaustive 2,598,960 five-card category verification."""
import itertools
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from evaluator import evaluate, reference_five

counts = [0] * 9
for cards in itertools.combinations(range(52), 5):
    value = evaluate(cards)
    assert value == reference_five(cards), cards
    counts[value >> 20] += 1
expected = [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40]
assert counts == expected, counts
print('Exhaustive five-card ranks and category counts passed:', counts)
