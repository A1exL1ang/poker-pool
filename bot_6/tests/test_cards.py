import random
import unittest
from bot6.cards import evaluate, reference, parse_card, card_name, hand_class


class CardsTest(unittest.TestCase):
    def test_parse_roundtrip(self):
        for c in range(52):
            self.assertEqual(parse_card(card_name(c)), c)
        self.assertEqual(parse_card("10S"), parse_card("Ts"))
        for card in (-1, 52, True, "1c", "Ace", "XX"):
            with self.assertRaises(ValueError):
                parse_card(card)

    def test_categories_and_kickers(self):
        examples = [
            ("As Kd 9h 7c 4s 3d 2c", 0),
            ("As Ad Kh 7c 4s 3d 2c", 1),
            ("As Ad Kh Kc 4s 3d 2c", 2),
            ("As Ad Ah Kc 9s 3d 2c", 3),
            ("As 2d 3h 4c 5s 9d Tc", 4),
            ("As Js 9s 5s 2s Kd Qh", 5),
            ("As Ad Ah Kc Ks Kd 2c", 6),
            ("As Ad Ah Ac Ks Kd 2c", 7),
            ("As Ks Qs Js Ts 2c 3d", 8),
        ]
        for cards, category in examples:
            self.assertEqual(evaluate(cards.split()) >> 24, category)
        self.assertEqual(evaluate("As Ad Ah Kc Qs 9d 8c".split()), evaluate("Ac Ad Ah Kd Qc 3s 2h".split()))
        self.assertEqual(evaluate("As Ad Kh Qc Js 9d 8c".split()), evaluate("Ac Ad Kd Qs Jh 3s 2h".split()))
        self.assertGreater(evaluate("6s 5d 4h 3c 2s".split()), evaluate("As 2d 3h 4c 5s".split()))

    def test_differential_reference(self):
        rng = random.Random(6026)
        for n, count in ((5, 1000), (6, 1500), (7, 5000)):
            for _ in range(count):
                cards = rng.sample(range(52), n)
                self.assertEqual(evaluate(cards), reference(cards), cards)

    def test_suit_invariance(self):
        rng = random.Random(77)
        for _ in range(500):
            cards = rng.sample(range(52), 7)
            suits = rng.sample(range(4), 4)
            changed = [c // 4 * 4 + suits[c % 4] for c in cards]
            self.assertEqual(evaluate(cards), evaluate(changed))

    def test_invalid_hand(self):
        for cards in ([0, 0, 1, 2, 3], [0, 1, 2, 3], list(range(8))):
            with self.assertRaises(ValueError):
                evaluate(cards)


if __name__ == "__main__":
    unittest.main()
