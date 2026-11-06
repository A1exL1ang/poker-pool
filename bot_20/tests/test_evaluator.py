"""Independent combinatorial checks for the exact rank implementation."""

import itertools
import random
import unittest
from collections import Counter

from pokerbot.evaluator import evaluate, card_id, card_text
from pokerbot.features import extract_features, FEATURE_NAMES


def reference_five(cards):
    counts = Counter(card // 4 + 2 for card in cards)
    grouped = sorted(((n, r) for r, n in counts.items()), reverse=True)
    ranks = sorted(counts, reverse=True)
    flush = len({card & 3 for card in cards}) == 1
    straight = 5 if ranks == [14, 5, 4, 3, 2] else (ranks[0] if len(ranks) == 5 and ranks[0] - ranks[-1] == 4 else 0)
    if flush and straight:
        cat, values = 8, [straight]
    elif grouped[0][0] == 4:
        cat, values = 7, [grouped[0][1], grouped[1][1]]
    elif [n for n, _ in grouped] == [3, 2]:
        cat, values = 6, [grouped[0][1], grouped[1][1]]
    elif flush:
        cat, values = 5, ranks
    elif straight:
        cat, values = 4, [straight]
    elif grouped[0][0] == 3:
        cat, values = 3, [r for _, r in grouped]
    elif [n for n, _ in grouped[:2]] == [2, 2]:
        cat, values = 2, [r for _, r in grouped]
    elif grouped[0][0] == 2:
        cat, values = 1, [r for _, r in grouped]
    else:
        cat, values = 0, ranks
    return (cat << 20) | sum(r << (16 - 4 * i) for i, r in enumerate(values))


class EvaluatorTests(unittest.TestCase):
    def test_known_ranks(self):
        samples = [
            ("Ac Kc Qc Jc Tc 2d 3d", 8, 14),
            ("Ac Ad Ah As Kd Qd Jd", 7, 14),
            ("Ac Ad Ah Ks Kd Kh 2s", 6, 14),
            ("Ac Qc 9c 5c 2c Kd 3h", 5, 14),
            ("Ac 2s 3d 4c 5s Kd Qd", 4, 5),
            ("Ac Ad Ah Ks Qd Jd 2c", 3, 14),
            ("Ac Ad Ks Kd Qs Qd 2h", 2, 14),
            ("Ac Ad Ks Qd Jd 8c 2h", 1, 14),
            ("Ac Ks Qd 9h 6d 4c 2h", 0, 14),
        ]
        for text, category, first in samples:
            value = evaluate([card_id(x) for x in text.split()])
            self.assertEqual(value >> 20, category, text)
            self.assertEqual((value >> 16) & 15, first, text)

    def test_against_independent_five_card_enumeration(self):
        rng = random.Random(792012)
        for size in (5, 6, 7):
            for _ in range(2000):
                cards = rng.sample(range(52), size)
                reference = max(reference_five(c) for c in itertools.combinations(cards, 5))
                self.assertEqual(evaluate(cards), reference, cards)

    def test_suit_and_order_invariance(self):
        rng = random.Random(7822)
        for _ in range(100):
            cards = rng.sample(range(52), 7)
            expected = evaluate(cards)
            for perm in itertools.permutations(range(4)):
                transformed = [(c & ~3) + perm[c & 3] for c in cards]
                rng.shuffle(transformed)
                self.assertEqual(evaluate(transformed), expected)

    def test_roundtrip_and_features(self):
        for c in range(52):
            self.assertEqual(card_id(card_text(c)), c)
        rng = random.Random(812)
        for size in (3, 4, 5):
            for _ in range(200):
                cards = rng.sample(range(52), size + 2)
                features = extract_features(cards[:2], cards[2:])
                self.assertEqual(len(features), len(FEATURE_NAMES))
                self.assertEqual(features, extract_features(cards[:2][::-1], cards[2:]))


if __name__ == "__main__":
    unittest.main()
