import itertools
import random
import unittest

from evaluator import evaluate, reference_five, hand_features


def cards(text):
    return ["23456789TJQKA".index(c[0]) * 4 + "cdhs".index(c[1]) for c in text.split()]


class EvaluatorTests(unittest.TestCase):
    def test_wheel_and_kickers(self):
        self.assertLess(evaluate(cards('Ac 2d 3h 4s 5c')), evaluate(cards('2c 3d 4h 5s 6c')))
        self.assertEqual(evaluate(cards('Ac Kc Qc Jc Tc 2d 3h')) >> 20, 8)
        self.assertEqual(evaluate(cards('Ac Ad Ah Kc Kd Kh Qh')), evaluate(cards('Ac Ad Ah Kc Kd')))
        self.assertGreater(evaluate(cards('Ac Ad Kh Qc Jd')), evaluate(cards('Ac Ad Kh Qc Td')))
        self.assertEqual(evaluate(cards('Ac Ad Ah As Kc Kd Kh')), evaluate(cards('Ac Ad Ah As Kc')))

    def test_random_reference(self):
        rng = random.Random(9217)
        for n in (5, 6, 7):
            for _ in range(5000):
                hand = rng.sample(range(52), n)
                expected = max(reference_five(c) for c in itertools.combinations(hand, 5))
                self.assertEqual(evaluate(hand), expected, hand)

    def test_suit_invariance(self):
        rng = random.Random(881)
        for _ in range(1000):
            hand = rng.sample(range(52), 7)
            permutation = rng.sample(range(4), 4)
            self.assertEqual(evaluate(hand), evaluate([c // 4 * 4 + permutation[c % 4] for c in hand]))

    def test_board_and_draws(self):
        f = hand_features(cards('Ac Kc'), cards('Qc 9c 2d'))
        self.assertTrue(f['flush_draw'])
        self.assertTrue(f['nut_flush_draw'])
        f = hand_features(cards('Ac Kd'), cards('Qc Jh Ts'))
        self.assertEqual(f['category'], 4)
        f = hand_features(cards('2d 3h'), cards('Ac Kc Qc Jc Tc'))
        self.assertTrue(f['plays_board'])


if __name__ == '__main__':
    unittest.main()
