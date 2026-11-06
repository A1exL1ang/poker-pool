import itertools
import random
import unittest

from pokerbot.evaluator import describe, evaluate, evaluate5, parse_card, parse_cards, validate_cards


class EvaluatorTests(unittest.TestCase):
    def test_known_categories_and_kickers(self):
        hands = [
            ("As Jd 9h 6c 2s", 0, (14, 11, 9, 6, 2)),
            ("As Ad Kh 9c 2s", 1, (14, 13, 9, 2)),
            ("As Ad Kh Kc 2s", 2, (14, 13, 2)),
            ("As Ad Ah Kc 2s", 3, (14, 13, 2)),
            ("As 2d 3h 4c 5s", 4, (5,)),
            ("As Js 9s 6s 2s", 5, (14, 11, 9, 6, 2)),
            ("As Ad Ah Kc Ks", 6, (14, 13)),
            ("As Ad Ah Ac 2s", 7, (14, 2)),
            ("As Ks Qs Js Ts", 8, (14,)),
        ]
        scores = []
        for text, category, kickers in hands:
            cards = parse_cards(text)
            score = evaluate(cards)
            self.assertEqual(score, evaluate5(cards), text)
            self.assertEqual(score >> 20, category, text)
            self.assertEqual(describe(score)[1], kickers, text)
            scores.append(score)
        self.assertEqual(scores, sorted(scores))

    def test_seven_card_tricky_cases(self):
        cases = [
            ("As Ad Ah Ks Kd Kh 2c", "As Ad Ah Ks Kd"),  # two trips
            ("As Ad Ks Kd Qs Qd 2c", "As Ad Ks Kd Qs"),  # three pairs
            ("As Ks Qs Js 9s 8s 2d", "As Ks Qs Js 9s"),  # six-card flush
            ("As 2s 3s 4s 5s 6d Kh", "As 2s 3s 4s 5s"),  # wheel straight flush
            ("As 2d 3h 4c 5s 6d 7h", "3h 4c 5s 6d 7h"),  # choose highest straight
            ("As Ad Ah Ac Ks Kd Qh", "As Ad Ah Ac Ks"),
            ("As Ad Ah Ks Qs Js Ts", "As Ks Qs Js Ts"),  # flush beats trips
            ("2s 3s 4s 5s 6s 6d 6h", "2s 3s 4s 5s 6s"),
            ("As Ad Ah Ks Kd Qs Qd", "As Ad Ah Ks Kd"),
        ]
        for full, best in cases:
            self.assertEqual(evaluate(parse_cards(full)), evaluate5(parse_cards(best)), full)

    def test_board_ties_and_private_kickers(self):
        board = parse_cards("As Ks Qs Js Ts")
        self.assertEqual(evaluate(board + parse_cards("2d 2h")), evaluate(board + parse_cards("Ad Ac")))
        board = parse_cards("Ac Ad Kh Qs 2c")
        self.assertGreater(evaluate(board + parse_cards("Kc 3d")), evaluate(board + parse_cards("Qh 3c")))

    def test_random_against_independent_five_card_reference(self):
        rng = random.Random(11001)
        for size in (5, 6, 7):
            for _ in range(1600):
                cards = rng.sample(range(52), size)
                reference = max(map(evaluate5, itertools.combinations(cards, 5)))
                self.assertEqual(evaluate(cards), reference, cards)

    def test_suit_and_order_invariance(self):
        rng = random.Random(11002)
        permutation = (2, 0, 3, 1)
        for _ in range(500):
            cards = rng.sample(range(52), 7)
            original = evaluate(cards)
            rng.shuffle(cards)
            self.assertEqual(evaluate(cards), original)
            transformed = [4 * (card // 4) + permutation[card % 4] for card in cards]
            self.assertEqual(evaluate(transformed), original)

    def test_input_boundaries(self):
        self.assertEqual(parse_card("10H"), parse_card("Th"))
        self.assertEqual(parse_card(" as "), 51)
        self.assertEqual(parse_cards("2c 2d 2h 2s"), (0, 1, 2, 3))
        for bad in ("1s", "xx", 52, -1, True, None):
            with self.assertRaises(ValueError):
                parse_card(bad)
        with self.assertRaises(ValueError):
            validate_cards([0, 0])
        for size in (0, 4, 8):
            with self.assertRaises(ValueError):
                evaluate(list(range(size)))


if __name__ == "__main__":
    unittest.main()
