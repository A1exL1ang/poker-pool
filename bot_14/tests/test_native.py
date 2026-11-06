"""Optional native-backend tests; portable fallback works without a binary."""

import ctypes
import itertools
import random
import unittest

from pokerbot.evaluator import card_id, evaluate
from pokerbot.native import NativeEquity


def cards(text):
    return [card_id(card) for card in text.split()]


def class_id(a, b):
    high, low = max(a // 4, b // 4), min(a // 4, b // 4)
    if high == low:
        return (12 - high) * 14
    return (12 - high) * 13 + 12 - low if a % 4 == b % 4 else (12 - low) * 13 + 12 - high


def reference_river(hole, board, percentiles, q=1, m=1):
    known = set(hole) | set(board)
    hero = evaluate(hole + board)
    ranks = sorted(evaluate([a, b] + board)
                   for a, b in itertools.combinations((c for c in range(52) if c not in known), 2)
                   if percentiles[class_id(a, b)] <= q)
    total = wins = 0.0
    i = 0
    while i < len(ranks):
        j = i + 1
        while j < len(ranks) and ranks[j] == ranks[i]:
            j += 1
        percentile = (i + j) / 2 / len(ranks)
        weight = 1 if m >= .999 or percentile >= 1 - m else .1
        count = (j - i) * weight
        total += count
        wins += count * (1 if hero > ranks[i] else .5 if hero == ranks[i] else 0)
        i = j
    return wins / total


class NativeFallbackTest(unittest.TestCase):
    def test_absent_binary_falls_back(self):
        backend = NativeEquity([.5] * 169, path=".control/nonexistent-bot14-native-library")
        self.assertFalse(backend.available)
        self.assertIsNone(backend.estimate([0, 1], [2, 3, 4]))
        self.assertEqual(backend.fallbacks, 1)


class NativeEquityTest(unittest.TestCase):
    def setUp(self):
        self.percentiles = [(i + .5) / 169 for i in range(169)]
        self.backend = NativeEquity(self.percentiles)
        if not self.backend.available:
            self.skipTest(self.backend.error)

    def tearDown(self):
        # The C ABI intentionally has process-global immutable deployment
        # tables. Restore real startup data after synthetic test fixtures so
        # a shared-process strategy benchmark cannot inherit test percentiles.
        from pokerbot.preflop import TABLES
        from pokerbot.model import MODEL
        restored = NativeEquity(TABLES.percentile)
        if restored.available and MODEL.loaded:
            restored.configure_model(MODEL.models, MODEL.feature_count)

    def test_exact_uniform_river_against_independent_python_enumeration(self):
        rng = random.Random(140901)
        for _ in range(10):
            hand = rng.sample(range(52), 7)
            expected = reference_river(hand[:2], hand[2:], self.percentiles)
            self.assertAlmostEqual(self.backend.estimate(hand[:2], hand[2:]), expected, places=12)

    def test_exact_weighted_range_river(self):
        hole, board = cards("As Qd"), cards("Ah Jh 7c 3s 2d")
        for q, m in ((1, .2), (.6, .4), (.2, .1), (.08, 1)):
            expected = reference_river(hole, board, self.percentiles, q, m)
            self.assertAlmostEqual(self.backend.estimate(hole, board, q=q, m=m), expected, places=12)

    def test_batched_river_matches_individual_calls(self):
        hole, board = cards("Ks Qh"), cards("Kd 8s 7c 3s 2d")
        cutoffs = (1.0, .7, .4, .1)
        batch = self.backend.river_equities(hole, board, q=.8, m_values=cutoffs)
        for actual, m in zip(batch, cutoffs):
            self.assertAlmostEqual(actual, self.backend.estimate(hole, board, q=.8, m=m), places=12)

    def test_board_royal_ties_multiway(self):
        hole, board = cards("2d 3s"), cards("Ac Kc Qc Jc Tc")
        self.assertEqual(self.backend.estimate(hole, board), .5)
        self.assertAlmostEqual(self.backend.estimate(hole, board, opponents=7, samples=64), 1 / 8)
        self.assertAlmostEqual(self.backend.river_iid(hole, board, opponents=7), 1 / 8)

    def test_rollouts_are_seeded_and_bounded(self):
        hole, board = cards("As Ks"), cards("Qs Js 2h")
        a = self.backend.estimate(hole, board, opponents=2, q=1, m=.4, seed=1400, samples=64)
        b = self.backend.estimate(hole, board, opponents=2, q=1, m=.4, seed=1400, samples=64)
        self.assertIsNotNone(a)
        self.assertEqual(a, b)
        self.assertEqual(self.backend.diagnostics()["trials"], 64)

    def test_impossible_joint_range_falls_back(self):
        percentiles = [1.0] * 169
        percentiles[0] = .01  # AA only; hero holds two aces, leaving one combo.
        backend = NativeEquity(percentiles)
        result = backend.estimate(cards("As Ad"), cards("2h 3c 4d"), opponents=2, q=.1, samples=16)
        self.assertIsNone(result)
        self.assertEqual(backend.diagnostics()["status"], 4)
        self.assertEqual(backend.diagnostics()["trials"], 0)

    def test_invalid_inputs_fall_back(self):
        self.assertIsNone(self.backend.estimate([0, 0], [1, 2, 3]))
        self.assertIsNone(self.backend.estimate([0, 1], [1, 2, 3]))
        self.assertIsNone(self.backend.estimate([0, 1], [2, 3, 4], q=0))
        self.assertIsNone(self.backend.estimate([0, 1], [2, 3, 4], opponents=8))
        self.assertIsNone(self.backend.estimate([0, 1], [2, 3, 4], samples=1000000))

    def test_iid_single_opponent_equals_exact_river(self):
        hole, board = cards("9s 8s"), cards("9d 8h 7c 6s 2d")
        expected = self.backend.estimate(hole, board, q=.8, m=.3)
        self.assertAlmostEqual(self.backend.river_iid(hole, board, opponents=1, q=.8, m=.3), expected, places=12)

    def test_actual_model_prediction_cpp_python_parity(self):
        from pokerbot.equity import feature_vector
        from pokerbot.model import EquityModel
        model = EquityModel()
        if not model.loaded:
            self.skipTest("optional offline model asset absent")
        if not self.backend.configure_model(model.models, model.feature_count):
            self.skipTest(self.backend.model_error)
        rng = random.Random(141107)
        for i in range(1000):
            hand = rng.sample(range(52), 5 + i % 3)
            f = feature_vector(hand[:2], hand[2:])
            q, m, opponents = rng.uniform(-.1, 1.1), rng.uniform(-.1, 1.1), rng.randrange(0, 9)
            actual = self.backend.predict_model(f, q, m, opponents)
            expected = model.predict(f, q, m, opponents)
            self.assertIsNotNone(actual)
            self.assertAlmostEqual(actual, expected, places=14)

    def test_augmented58_feature_model(self):
        from pokerbot.derived import augment
        from pokerbot.equity import feature_vector
        models = {str(street): {"bias": .2, "trees": [
            [[57, .5, 1, 2], [-1, -.1], [-1, .3]],
            [[44, .5, 1, 2], [-1, .0], [-1, .1]],
        ]} for street in (3, 4, 5)}
        if not self.backend.configure_model(models, 58):
            self.skipTest(self.backend.model_error)
        hole, board = cards("As Qd"), cards("Ah Jh 7c 3s 2d")
        f = feature_vector(hole, board)
        for q in (.4, .9):
            vector = augment(f, q, .3, 1)
            expected = .2 + (-.1 if vector[57] <= .5 else .3) + (0 if q <= .5 else .1)
            self.assertAlmostEqual(self.backend.predict_model(f, q, .3, 1), expected)
            self.assertAlmostEqual(self.backend.predict_model(vector), expected)

    def test_invalid_model_rejected_without_disabling_equity(self):
        cyclic = {"3": {"bias": .5, "trees": [[[0, 4, 0, 1], [-1, .1]]]}}
        self.assertFalse(self.backend.configure_model(cyclic, 47))
        self.assertIsNone(self.backend.predict_model([0] * 44))
        self.assertIsNotNone(self.backend.estimate(cards("As Qd"), cards("Ah Jh 7c 3s 2d")))


if __name__ == "__main__":
    unittest.main()
