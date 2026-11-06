"""Independent rank reference, edge cases and randomized 5/6/7-card checks."""
import itertools
import os
import random
import unittest
from collections import Counter

from cards import card_name, evaluate, parse_card, preflop_key


def reference_five(cards):
    ranks = sorted((c // 4 for c in cards), reverse=True)
    groups = sorted(((n, r) for r, n in Counter(ranks).items()), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    straight = -1
    if len(set(ranks)) == 5:
        if ranks[0] - ranks[-1] == 4:
            straight = ranks[0]
        elif ranks == [12, 3, 2, 1, 0]:
            straight = 3
    if flush and straight >= 0:
        category, kickers = 8, [straight]
    elif groups[0][0] == 4:
        category, kickers = 7, [groups[0][1], groups[1][1]]
    elif [g[0] for g in groups] == [3, 2]:
        category, kickers = 6, [g[1] for g in groups]
    elif flush:
        category, kickers = 5, ranks
    elif straight >= 0:
        category, kickers = 4, [straight]
    elif groups[0][0] == 3:
        category, kickers = 3, [g[1] for g in groups]
    elif [g[0] for g in groups[:2]] == [2, 2]:
        category, kickers = 2, [g[1] for g in groups]
    elif groups[0][0] == 2:
        category, kickers = 1, [g[1] for g in groups]
    else:
        category, kickers = 0, ranks
    out = category << 20
    for i, rank in enumerate(kickers):
        out |= rank << (16 - 4 * i)
    return out


class CardsTest(unittest.TestCase):
    def test_card_encoding(self):
        for card in range(52):
            self.assertEqual(parse_card(card_name(card)), card)
        self.assertEqual(parse_card("10H"), parse_card("Th"))
        for bad in (-1, 52, "XX", "A", "11h"):
            with self.assertRaises(ValueError):
                parse_card(bad)

    def test_preflop_classes(self):
        self.assertEqual(preflop_key(parse_card("As"), parse_card("Ks")), "AKs")
        self.assertEqual(preflop_key(parse_card("Kd"), parse_card("As")), "AKo")
        self.assertEqual(preflop_key(parse_card("As"), parse_card("Ac")), "AA")
        classes = Counter(preflop_key(pair) for pair in itertools.combinations(range(52), 2))
        self.assertEqual(len(classes), 169)
        self.assertEqual(sorted(Counter(classes.values()).items()), [(4, 78), (6, 13), (12, 78)])

    def test_edge_cases(self):
        examples = [
            ("As 2s 3s 4s 5s Kd Kh", 8),
            ("As Ad Ac Ah Ks Kd Kc", 7),
            ("As Ad Ac Ks Kd Kc 2h", 6),
            ("As Ks Qs 9s 8s 4s 3s", 5),
            ("As 2d 3c 4h 5s 6d 7c", 4),
            ("As Ad Ac Kh Qs 9d 2c", 3),
            ("As Ad Ks Kh Qs Qd 2c", 2),
            ("As Ad Ks Qh Js 9d 2c", 1),
            ("As Kd Js 9h 7s 4d 2c", 0),
        ]
        for text, cat in examples:
            cards = [parse_card(c) for c in text.split()]
            value = evaluate(cards)
            self.assertEqual(value >> 20, cat, text)
            self.assertEqual(value, max(map(reference_five, itertools.combinations(cards, 5))), text)

    def test_random_against_reference(self):
        rng = random.Random(617308)
        for n, count in ((5, 5000), (6, 2000), (7, 3000)):
            for _ in range(count):
                cards = rng.sample(range(52), n)
                self.assertEqual(evaluate(cards), max(map(reference_five, itertools.combinations(cards, 5))), cards)

    def test_suit_and_order_invariance(self):
        rng = random.Random(1941)
        for _ in range(1000):
            cards = rng.sample(range(52), 7)
            permutation = rng.sample(range(4), 4)
            other = [(c // 4) * 4 + permutation[c % 4] for c in reversed(cards)]
            self.assertEqual(evaluate(cards), evaluate(other))

    @unittest.skipUnless(os.environ.get("TEST_EXHAUSTIVE") == "1", "set TEST_EXHAUSTIVE=1 for all 2,598,960 five-card hands")
    def test_exhaustive_five_card_frequencies(self):
        counts = Counter(evaluate(cards) >> 20 for cards in itertools.combinations(range(52), 5))
        self.assertEqual([counts[i] for i in range(9)],
                         [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40])


if __name__ == "__main__":
    unittest.main()
