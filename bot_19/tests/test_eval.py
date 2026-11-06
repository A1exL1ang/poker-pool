import random
import unittest

from poker_eval import FEATURE_NAMES, evaluate, features, reference_evaluate


def cards(text):
    rs, ss = "23456789TJQKA", "cdhs"
    return [rs.index(c[0]) * 4 + ss.index(c[1]) for c in text.split()]


class EvaluatorTests(unittest.TestCase):
    def test_known_categories_and_kickers(self):
        hands = [
            "Ac Kd Jh 9s 7c 4d 2h",
            "Ac Ad Kh Qs 9c 7d 2h",
            "Ac Ad Kh Ks 9c 9d 2h",
            "Ac Ad Ah Ks Qc 7d 2h",
            "Ac 2d 3h 4s 5c 7d 9h",
            "Ac Kc Jc 9c 7c 4d 2h",
            "Ac Ad Ah Ks Kc Qd Qh",
            "Ac Ad Ah As Kc Qd 2h",
            "9c Tc Jc Qc Kc 4d 2h",
        ]
        scores = [evaluate(cards(h)) for h in hands]
        self.assertEqual([s >> 20 for s in scores], list(range(9)))
        for h in hands:
            self.assertEqual(evaluate(cards(h)), reference_evaluate(cards(h)))
        self.assertEqual((scores[4] >> 16) & 15, 5)

    def test_two_triplets_full_house(self):
        h = cards("Ac Ad Ah Ks Kc Kd 2h")
        self.assertEqual(evaluate(h), reference_evaluate(h))

    def test_board_plays_and_three_pairs(self):
        board = cards("Ac Kc Qc Jc Tc")
        self.assertEqual(evaluate(board + cards("2h 3h")), evaluate(board + cards("As Ad")))
        h = cards("Ac Ad Kh Ks Qc Qd Jh")
        self.assertEqual(evaluate(h), reference_evaluate(h))

    def test_random_reference(self):
        rng = random.Random(190190)
        for count in (5, 6, 7):
            for _ in range(3000):
                hand = rng.sample(range(52), count)
                self.assertEqual(evaluate(hand), reference_evaluate(hand), hand)

    def test_order_and_suit_invariance(self):
        rng = random.Random(9319)
        for _ in range(300):
            hand = rng.sample(range(52), 7)
            a = evaluate(hand)
            rng.shuffle(hand)
            self.assertEqual(a, evaluate(hand))
            self.assertEqual(a, evaluate([4 * (c // 4) + ((c % 4 + 1) % 4) for c in hand]))

    def test_feature_schema_and_suit_invariance(self):
        rng = random.Random(5319)
        for size in (3, 4, 5):
            for _ in range(100):
                hand = rng.sample(range(52), size + 2)
                f = features(hand[:2], hand[2:], .25, .8)
                self.assertEqual(len(f), len(FEATURE_NAMES))
                rotated = [4 * (c // 4) + ((c % 4 + 1) % 4) for c in hand]
                self.assertEqual(f, features(rotated[:2], rotated[2:], .25, .8))


if __name__ == "__main__":
    unittest.main()
