import random
import unittest
from pokerbot.evaluator import evaluate, reference, parse_card, card_name
from pokerbot.equity import hand_class, preflop_equity, EquityEstimator


def cards(s):
    return [parse_card(x) for x in s.split()]


class EvaluatorTests(unittest.TestCase):
    def test_card_roundtrip(self):
        for c in range(52):
            self.assertEqual(c, parse_card(card_name(c)))
        self.assertEqual(parse_card("10h"), parse_card("Th"))

    def test_categories_and_kickers(self):
        cases = ["As Ks Qs Js Ts 2d 3c", "As Ah Ad Ac Ks 2h 3c",
                 "As Ah Ad Ks Kh Kd 2c", "As Js 9s 5s 2s Kd Qd",
                 "As 2h 3d 4c 5s Kd Qd", "As Ah Ad Ks Qh 2d 3d",
                 "As Ah Ks Kh Qd Qh 2c", "As Ah Ks Qh Jd 2h 3d",
                 "As Kh Qd Jc 9s 2h 3d"]
        for category, s in zip(range(8, -1, -1), cases):
            with self.subTest(category=category):
                hand = cards(s)
                self.assertEqual(evaluate(hand) >> 24, category)
                self.assertEqual(evaluate(hand), reference(hand))
        self.assertGreater(evaluate(cards("As 2h 3d 4c 6s 5d Qd")), evaluate(cards("As 2h 3d 4c 5s Kd Qd")))
        self.assertGreater(evaluate(cards("As Ah Ks Qh Jd 2h 3d")), evaluate(cards("As Ah Ks Qh Td 2h 3d")))

    def test_independent_random_reference(self):
        rng = random.Random(706106)
        for size, count in ((5, 12000), (6, 4000), (7, 6000)):
            for _ in range(count):
                hand = rng.sample(range(52), size)
                self.assertEqual(evaluate(hand), reference(hand), [card_name(c) for c in hand])

    def test_permutation_suit_invariance(self):
        rng = random.Random(102)
        for _ in range(500):
            hand = rng.sample(range(52), 7)
            suits = rng.sample(range(4), 4)
            changed = [suits[c // 13] * 13 + c % 13 for c in hand]
            self.assertEqual(evaluate(hand), evaluate(changed))
            rng.shuffle(changed)
            self.assertEqual(evaluate(hand), evaluate(changed))

    def test_input_validation(self):
        for bad in ("1s", "Az", 52, -1, None, True):
            with self.assertRaises((ValueError, TypeError)):
                parse_card(bad)
        with self.assertRaises(ValueError):
            evaluate([0, 0, 1, 2, 3])

    def test_preflop_sanity(self):
        aa = hand_class(*cards("As Ah"))
        junk = hand_class(*cards("7s 2h"))
        self.assertGreater(preflop_equity(aa, 1), .84)
        self.assertLess(preflop_equity(aa, 1), .87)
        self.assertLess(preflop_equity(junk, 1), .36)
        self.assertGreater(preflop_equity(aa, .1), preflop_equity(junk, .1))
        self.assertGreater(preflop_equity(aa, 1, 2), preflop_equity(aa, 1, 7))

    def test_estimator_extremes_and_determinism(self):
        a = EquityEstimator(1)
        b = EquityEstimator(1)
        hole, board = cards("As Ks"), cards("Qs Js Ts 2d 3h")
        e = a.estimate(hole, board, [.25])
        self.assertEqual(e, b.estimate(hole, board, [.25]))
        self.assertGreater(e[0], .97)
        hole, board = cards("2c 3c"), cards("As Ks Qs Js Ts")
        self.assertAlmostEqual(a.estimate(hole, board, [1])[0], .5)
        self.assertAlmostEqual(a.estimate(hole, board, [1, 1])[0], 1/3, delta=.015)


if __name__ == "__main__":
    unittest.main()
