import itertools
import random
import unittest

from pokerbot.equity import EquityEstimator, hand_class, preflop_percentile, _compile_trees
from pokerbot.evaluator import card_id
from pokerbot.features import extract_features, FEATURE_INDEX


def cards(text):
    return tuple(card_id(c) for c in text.split())


class EquityTests(unittest.TestCase):
    def test_compiler_rejects_non_trees(self):
        with self.assertRaises(ValueError):
            _compile_trees(0, (((0, .5, 1, 1, 0), (-1, 0, 0, 0, .25)),), 68)
        with self.assertRaises(ValueError):
            _compile_trees(0, (((0, .5, 0, 0, 0),),), 68)
        with self.assertRaises(ValueError):
            _compile_trees(0, (((-1, 0, 0, 0, float("nan")),),), 68)

    def test_hand_classes(self):
        represented = {hand_class((a, b)) for a in range(52) for b in range(a + 1, 52)}
        self.assertEqual(represented, set(range(169)))
        self.assertEqual(hand_class(cards("As Kd")), hand_class(cards("Ah Kc")))
        self.assertNotEqual(hand_class(cards("As Ks")), hand_class(cards("As Kd")))

    def test_order(self):
        self.assertLess(preflop_percentile(cards("As Ah")), preflop_percentile(cards("2s 7h")))

    def test_tight_preflop_table_boundaries(self):
        model = EquityEstimator()
        if not model.preflop_loaded or model.preflop["fractions"][0] > .02:
            self.skipTest("Expanded preflop table not present")
        kings, queens = cards("Ks Kh"), cards("Qs Qh")
        self.assertLess(model.estimate(kings, (), 1, .02), model.estimate(kings, (), 1, .05))
        self.assertLess(model.estimate(queens, (), 1, .02), model.estimate(kings, (), 1, .02))
        self.assertEqual(model.estimate(kings, (), 1, .001), model.estimate(kings, (), 1, .02))
        for hand in (kings, queens):
            lo, hi = model.estimate(hand, (), 1, .02), model.estimate(hand, (), 1, .03)
            middle = model.estimate(hand, (), 1, .025)
            self.assertGreaterEqual(middle, min(lo, hi))
            self.assertLessEqual(middle, max(lo, hi))

    def test_royals_and_empty_opponents(self):
        model = EquityEstimator()
        self.assertEqual(model.estimate(cards("As Ks"), cards("Qs Js Ts 2d 3c"), 7), 1)
        self.assertEqual(model.estimate(cards("2d 3c"), cards("As Ks Qs Js Ts"), 7), .125)
        self.assertEqual(model.estimate(cards("2d 3c"), cards("As Ks Qs Js Ts"), 0), 1)

    def test_feature_suit_invariance(self):
        rng = random.Random(813218)
        for _ in range(100):
            sample = rng.sample(range(52), rng.choice((5, 6, 7)))
            before = extract_features(sample[:2], sample[2:])
            for permutation in itertools.permutations(range(4)):
                after_cards = [(c & ~3) + permutation[c & 3] for c in sample]
                after = extract_features(after_cards[:2], after_cards[2:])
                self.assertEqual(before, after)

    def test_relational_board_features(self):
        low_straight = extract_features(cards("2s Ad"), cards("4c 6s 5d 3c Ks"))
        self.assertEqual(low_straight[FEATURE_INDEX["possible_straight_high"]], 8)
        self.assertEqual(low_straight[FEATURE_INDEX["single_card_straight_high"]], 7)
        self.assertEqual(low_straight[FEATURE_INDEX["straight_gap_to_nuts"]], 2)
        self.assertEqual(low_straight[FEATURE_INDEX["single_card_higher_straight_outs"]], 4)
        underfull = extract_features(cards("9s 3s"), cards("8c 9d 3c 3h Td"))
        self.assertEqual(underfull[FEATURE_INDEX["higher_full_house_pocket_pair_combos"]], 7)
        self.assertEqual(underfull[FEATURE_INDEX["board_singletons_above_primary"]], 3)
        board_pairs = extract_features(cards("4d 7s"), cards("5c Ac 5s 4c Ad"))
        self.assertEqual(board_pairs[FEATURE_INDEX["first_kicker_difference"]], 2)
        self.assertEqual(board_pairs[FEATURE_INDEX["first_private_kicker"]], 7)

    def test_bounded_and_reused_features(self):
        rng = random.Random(177201)
        model = EquityEstimator()
        for _ in range(300):
            sample = rng.sample(range(52), rng.choice((5, 6, 7)))
            hole, board = sample[:2], sample[2:]
            n, fraction, pressure = rng.randrange(1, 8), rng.uniform(.05, 1), rng.random()
            prediction = model.estimate(hole, board, n, fraction, pressure)
            self.assertEqual(prediction, model.predict(extract_features(hole, board), n, fraction, pressure))
            self.assertGreaterEqual(prediction, 0)
            self.assertLessEqual(prediction, 1)

    def test_explicit_missing_model_fallback(self):
        model = EquityEstimator(model_path="nonexistent-bot20-model.gz", preflop_path="nonexistent-bot20-preflop.json")
        self.assertFalse(model.model_loaded)
        self.assertFalse(model.preflop_loaded)
        self.assertGreater(model.estimate(cards("As Ah"), (), 1), model.estimate(cards("2s 7h"), (), 1))

    def test_compiled_predictor_matches_interpreter(self):
        model = EquityEstimator()
        if not model.model_loaded:
            self.skipTest("Trained asset not present")
        rng = random.Random(512611)
        for _ in range(1000):
            sample = rng.sample(range(52), rng.choice((5, 6, 7)))
            f = extract_features(sample[:2], sample[2:])
            high, low = int(f[1]) - 2, int(f[2]) - 2
            index = high * 13 + low if f[4] else low * 13 + high
            vector = (*f[:model.feature_count], rng.randint(1, 7), rng.uniform(.05, 1), rng.random(), model.preflop["percentiles"][index])
            base, trees, predictor = model._models[int(f[0])]
            expected = base
            for tree in trees:
                node = tree[0]
                while node[0] >= 0:
                    node = tree[node[2] if vector[node[0]] <= node[1] else node[3]]
                expected += node[4]
            self.assertEqual(predictor(vector), expected)
        self.assertIs(model.trees, EquityEstimator().trees)


if __name__ == "__main__":
    unittest.main()
