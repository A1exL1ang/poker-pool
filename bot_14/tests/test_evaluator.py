import itertools
import random
import unittest

from pokerbot.evaluator import card_id, card_text, evaluate, evaluate_checked, reference_evaluate


def cards(text):
    return [card_id(card) for card in text.split()]


class EvaluatorTest(unittest.TestCase):
    def test_categories(self):
        examples = [
            "Ac Jd 9h 7s 4d 3c 2s",
            "Ac Ad 9h 7s 4d 3c 2s",
            "Ac Ad 9h 9s 4d 3c 2s",
            "Ac Ad Ah 7s 4d 3c 2s",
            "Ac 2d 3h 4s 5d 9c Ts",
            "Ac Jc 9c 7c 4c 3d 2s",
            "Ac Ad Ah 7s 7d 3c 2s",
            "Ac Ad Ah As 4d 3c 2s",
            "Ac 2c 3c 4c 5c 9d Ts",
        ]
        for category, hand in enumerate(examples):
            self.assertEqual(evaluate(cards(hand)) >> 20, category, hand)

    def test_wheel_loses_to_six_high(self):
        self.assertLess(evaluate(cards("Ac 2d 3h 4s 5d 9c Ts")), evaluate(cards("2c 3d 4h 5s 6d 9h Jc")))

    def test_board_tie(self):
        board = cards("Tc Jc Qc Kc Ac")
        self.assertEqual(evaluate(board + cards("2d 3h")), evaluate(board + cards("As Ad")))

    def test_kickers_ignore_sixth_and_seventh_card(self):
        self.assertEqual(evaluate(cards("Ac Ad Ah Kc Qc 3d 2s")), evaluate(cards("As Ac Ad Ks Qs 9d 8s")))
        self.assertLess(evaluate(cards("Ac Ad Ah Kc Jc 3d 2s")), evaluate(cards("As Ac Ad Ks Qs 9d 8s")))
        self.assertEqual(evaluate(cards("Ac Ad Kh Qc Jd 9s 8s")), evaluate(cards("Ac Ad Kc Qd Js 3h 2h")))

    def test_two_triplets_and_three_pairs(self):
        for hand in ("Ac Ad Ah Kc Kd Kh Qd", "Ac Ad Kc Kd Qc Qd Jc", "Ac Ad Ah As Kc Qc Jc"):
            self.assertEqual(evaluate(cards(hand)), reference_evaluate(cards(hand)))

    def test_roundtrip(self):
        self.assertEqual([card_id(card_text(i)) for i in range(52)], list(range(52)))

    def test_validation(self):
        for hand in ([0, 1], [0, 1, 2, 3, 3], [0, 1, 2, 3, 52], [0, 1, 2, 3, -1], [0, 1, 2, 3, True]):
            with self.assertRaises(ValueError):
                evaluate_checked(hand)

    def test_reference_random(self):
        rng = random.Random(140089)
        for length in (5, 6, 7):
            for _ in range(1200):
                hand = rng.sample(range(52), length)
                self.assertEqual(evaluate(hand), reference_evaluate(hand), hand)

    def test_suit_permutation_invariance(self):
        rng = random.Random(317)
        for _ in range(100):
            hand = rng.sample(range(52), 7)
            expected = evaluate(hand)
            for perm in itertools.permutations(range(4)):
                self.assertEqual(evaluate([card // 4 * 4 + perm[card % 4] for card in hand]), expected)


if __name__ == "__main__":
    unittest.main()
