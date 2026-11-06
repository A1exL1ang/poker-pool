import itertools
import random
import unittest

from pokerbot.evaluator import evaluate, parse_cards
from pokerbot.features import FEATURE_NAMES, features


def permute_suits(cards, permutation):
    return tuple(4 * (card // 4) + permutation[card % 4] for card in cards)


class FeatureTests(unittest.TestCase):
    def named(self, hole, board):
        values = features(parse_cards(hole), parse_cards(board))
        return dict(zip(FEATURE_NAMES, values))

    def test_shape_rank_encoding_and_named_indices(self):
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))
        self.assertEqual(FEATURE_NAMES[6], "made_category")
        self.assertEqual(FEATURE_NAMES[21], "board_max_suit")
        self.assertEqual(FEATURE_NAMES[32:34], ("best_flush_count", "flush_hole_high"))
        self.assertEqual(FEATURE_NAMES[36:39], ("best_straight_count", "best_straight_hole", "board_straight_count"))
        rng = random.Random(11031)
        for boardlen in (3, 4, 5):
            for _ in range(150):
                cards = rng.sample(range(52), boardlen + 2)
                hole, board = cards[:2], cards[2:]
                values = features(hole, board)
                self.assertEqual(len(values), len(FEATURE_NAMES))
                self.assertTrue(all(isinstance(value, int) for value in values))
                self.assertEqual(values[0], boardlen)
                self.assertEqual(values[1:3], sorted([c // 4 + 2 for c in hole], reverse=True))
                score = evaluate(cards)
                self.assertEqual(values[6], score >> 20)
                self.assertEqual(values[7:12], [(score >> k) & 15 for k in (16, 12, 8, 4, 0)])
                self.assertEqual(values[12:17], sorted([c // 4 + 2 for c in board], reverse=True) + [0] * (5 - boardlen))
                self.assertTrue(0 <= values[32] <= 7)
                self.assertTrue(0 <= values[36] <= 5)
                self.assertTrue(0 <= values[38] <= min(5, boardlen))

    def test_hole_and_board_order_invariance(self):
        rng = random.Random(11032)
        for _ in range(300):
            cards = rng.sample(range(52), rng.choice((5, 6, 7)))
            hole, board = cards[:2], cards[2:]
            original = features(hole, board)
            rng.shuffle(board)
            self.assertEqual(features(hole[::-1], board), original)

    def test_all_suit_permutations(self):
        rng = random.Random(11033)
        permutations = list(itertools.permutations(range(4)))
        cases = [(parse_cards("As Kd"), parse_cards("2s 2d 9h")),
                 (parse_cards("As Ad"), parse_cards("2s 2d 9h")),
                 (parse_cards("As Ad"), parse_cards("2s Kd 9d")),
                 (parse_cards("As Kh"), parse_cards("Qs Jh 2d 2c"))]
        for _ in range(160):
            cards = rng.sample(range(52), rng.choice((5, 6, 7)))
            cases.append((cards[:2], cards[2:]))
        for hole, board in cases:
            original = features(hole, board)
            for permutation in permutations:
                transformed = features(permute_suits(hole, permutation), permute_suits(board, permutation))
                self.assertEqual(transformed, original, (hole, board, permutation))

    def test_tied_flush_suits_prefer_owned_high_rank(self):
        values = self.named("As Kd", "2s 2d 9h")
        self.assertEqual(values["best_flush_count"], 2)
        self.assertEqual(values["flush_hole_high"], 14)
        self.assertEqual(values["flush_above"], 0)

    def test_owned_draw_and_wheel_features(self):
        values = self.named("As Ks", "Qs Js 2d")
        self.assertEqual(values["best_flush_count"], 4)
        self.assertEqual(values["flush_hole_high"], 14)
        self.assertEqual(values["flush_hole_low"], 13)
        self.assertEqual(values["best_straight_count"], 4)
        self.assertEqual(values["best_straight_hole"], 2)
        self.assertEqual(values["straight_out_ranks"], 1)
        self.assertEqual(values["straight_nut_high"], 14)
        values = self.named("As 2d", "3h 4c Ks")
        self.assertEqual(values["best_straight_count"], 4)
        self.assertEqual(values["best_straight_hole"], 2)
        self.assertEqual(values["straight_out_ranks"], 1)
        self.assertEqual(values["straight_nut_high"], 5)
        self.assertEqual(values["wheel_cards"], 4)

    def test_set_and_board_texture_features(self):
        values = self.named("7s 7h", "As 7d 2c")
        self.assertEqual(values["made_category"], 3)
        self.assertEqual(values["set_rank"], 7)
        self.assertEqual(values["board_distinct"], 3)
        values = self.named("As Kd", "2c 2d 2h 9s 9c")
        self.assertEqual(values["board_pairs"], 1)
        self.assertEqual(values["board_trips"], 1)
        self.assertEqual(values["made_category"], 6)


if __name__ == "__main__":
    unittest.main()
