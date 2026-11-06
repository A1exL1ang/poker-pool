"""Independent correctness checks for the original native evaluator/sampler."""
from __future__ import annotations
from collections import Counter
from itertools import combinations
import math
import random
import unittest

from bot13.native import NativeEvaluator, combo_index, native_available


def reference5(cards):
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    counts = Counter(ranks)
    grouped = sorted(((n, r) for r, n in counts.items()), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    distinct = sorted(counts, reverse=True)
    straight = 0
    if len(distinct) == 5 and distinct[0] - distinct[-1] == 4:
        straight = distinct[0]
    elif distinct == [14, 5, 4, 3, 2]:
        straight = 5
    if flush and straight:
        return (8, straight)
    if grouped[0][0] == 4:
        return (7, grouped[0][1], grouped[1][1])
    if [x[0] for x in grouped] == [3, 2]:
        return (6, grouped[0][1], grouped[1][1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if grouped[0][0] == 3:
        return (3, grouped[0][1], *sorted((r for r in ranks if r != grouped[0][1]), reverse=True))
    if [x[0] for x in grouped[:2]] == [2, 2]:
        return (2, *sorted((grouped[0][1], grouped[1][1]), reverse=True), grouped[2][1])
    if grouped[0][0] == 2:
        return (1, grouped[0][1], *sorted((r for r in ranks if r != grouped[0][1]), reverse=True))
    return (0, *ranks)


def encode_reference(rank):
    score = rank[0] << 24
    for i, r in enumerate(rank[1:]):
        score |= r << (20 - 4*i)
    return score


def reference(cards):
    return encode_reference(max(reference5(c) for c in combinations(cards, 5)))


@unittest.skipUnless(native_available, "build optional backend with python native/build.py")
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = NativeEvaluator()

    def test_known_categories(self):
        for cards in ([48, 44, 40, 36, 32], [48, 49, 50, 51, 0], [48, 49, 50, 44, 45],
                      [0, 8, 20, 32, 48], [0, 5, 10, 15, 16], [0, 1, 2, 8, 12],
                      [0, 1, 4, 5, 8], [0, 1, 8, 12, 16], [0, 5, 10, 20, 32],
                      [48, 0, 5, 10, 15, 21, 27], [0, 1, 2, 4, 5, 6, 48]):
            self.assertEqual(self.engine.evaluate(cards), reference(cards))

    def test_random_5_6_7_against_independent_combinatorial_reference(self):
        rng = random.Random(1306001)
        for n in (5, 6, 7):
            for _ in range(2500):
                cards = rng.sample(range(52), n)
                self.assertEqual(self.engine.evaluate(cards), reference(cards), cards)

    def test_full_five_card_category_histogram(self):
        self.assertEqual(self.engine.five_card_histogram(),
                         (1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40))

    def test_invalid_inputs(self):
        for cards in ([0, 0, 1, 2, 3], [-1, 0, 1, 2, 3], [0, 1, 2, 3, 52], [0, 1]):
            with self.assertRaises(ValueError):
                self.engine.evaluate(cards)
        with self.assertRaises(ValueError):
            self.engine.equity([0, 1], [0, 2, 3])
        with self.assertRaises(ValueError):
            self.engine.equity([0, 1], [], 1, [0.0]*1326)

    def test_combo_index_complete(self):
        self.assertEqual([combo_index(i, j) for i in range(52) for j in range(i+1, 52)], list(range(1326)))

    def test_reproducible_uniform_and_exact_river(self):
        first = self.engine.equity([48, 49], [], seed=130013, samples=20000)
        self.assertEqual(first, self.engine.equity([48, 49], [], seed=130013, samples=20000))
        self.assertAlmostEqual(first[0], 0.852, delta=0.016)
        board = [0, 5, 10, 15, 16]
        hero = [48, 49]
        ours = reference(hero+board)
        shares = []
        for opp in combinations((c for c in range(52) if c not in hero+board), 2):
            theirs = reference(list(opp)+board)
            shares.append(1 if ours > theirs else 0.5 if ours == theirs else 0)
        estimate, error = self.engine.equity(hero, board, samples=0)
        self.assertAlmostEqual(estimate, sum(shares)/len(shares), places=12)
        self.assertEqual(error, 0)

    def test_weighted_fixed_hand_and_board_tie(self):
        weights = [0.0]*1326
        weights[combo_index(44, 45)] = 1
        estimate, _ = self.engine.equity([48, 49], [], 1, weights, seed=1313, samples=20000)
        self.assertAlmostEqual(estimate, 0.819, delta=0.017)
        eq, error = self.engine.equity([0, 1], [32, 36, 40, 44, 48], 7, samples=100)
        self.assertEqual(eq, 1/8)
        self.assertEqual(error, 0)

    def test_weighted_whole_tuple_conditional_distribution(self):
        # Only three of four opponent tuples are card-disjoint. Conditioning the
        # second opponent alone would give the wrong .06/.44 joint equity.
        weights = [0.0]*(2*1326)
        weights[combo_index(16, 17)] = .8
        weights[combo_index(44, 45)] = .2
        weights[1326+combo_index(16, 18)] = .7
        weights[1326+combo_index(36, 37)] = .3
        equity, error = self.engine.equity([48, 49], [0, 5, 10, 15, 20], 2, weights, seed=130133, samples=30000)
        self.assertAlmostEqual(equity, .06/.44, delta=.008)
        self.assertLess(error, .003)

    def test_whole_tuple_rejection_and_prefixes(self):
        board = [32, 36, 40, 44, 48]
        data = self.engine.modeled_equity([0, 1], board, [(1, 0, .08, .03, .25)]*7, seed=13, samples=128)
        for n, value in enumerate(data["prefixes"], 1):
            self.assertAlmostEqual(value, 1/(n+1), places=12)
        self.assertEqual(data["equity"], 1/8)
        self.assertEqual(data["strength"], 0.5)
        self.assertGreater(data["accepted"], 100)

    def test_modeled_monotone_prefixes_and_features(self):
        data = self.engine.modeled_equity([48, 44], [40, 36, 1], [(0.5, .65, .08, .05, .3)]*3, seed=13, samples=256)
        self.assertEqual(data["flush_draw"], 1)
        self.assertGreater(data["straight_draw"], 0)
        self.assertEqual(data["prefixes"][-1], data["equity"])
        self.assertTrue(all(a >= b for a, b in zip(data["prefixes"], data["prefixes"][1:])))

    def test_native_private_straight_draw_excludes_lower_than_board(self):
        # A 3 makes 23456 using our deuce but also makes the superior 34567 on
        # board. The private-card straight is irrelevant to winning the pot.
        data = self.engine.modeled_equity([0, 45], [8, 13, 18, 23], [(1, 0, .08, .03, .25)], seed=13, samples=32)
        self.assertEqual(data["straight_draw"], 0)
        data = self.engine.modeled_equity([48, 49], [0, 5, 10, 15], [(1, 0, .08, .03, .25)], seed=13, samples=32)
        self.assertEqual(data["straight_draw"], 0)

    def test_native_icm_against_independent_all_players_dp(self):
        from lab.simulator import icm
        rng = random.Random(130919)
        for n in range(1, 9):
            for _ in range(12):
                stacks = [rng.randrange(1, 10000) if rng.random() > .25 else 0 for _ in range(n)]
                payouts = sorted((rng.random() for _ in range(n)), reverse=True)
                expected = icm(stacks, payouts)
                for hero in range(n):
                    self.assertAlmostEqual(self.engine.icm_equity(stacks, payouts, hero), expected[hero], places=11)

    def test_native_icm_special_cases(self):
        for hero in range(8):
            self.assertAlmostEqual(self.engine.icm_equity([1]*8, [10, 5, 3, 0, 0, 0, 0, 0], hero), 18/8)
            self.assertAlmostEqual(self.engine.icm_equity([1]*8, [1, 0, 0, 0, 0, 0, 0, 0], hero), 1/8)
            self.assertAlmostEqual(self.engine.icm_equity([0]*8, [10, 5, 3, 0, 0, 0, 0, 0], hero), 18/8)
        self.assertEqual(self.engine.icm_equity([1e300, 1, 1], [1, .5, 0], 1), .25)
        self.assertEqual(self.engine.icm_equity([10, 0, 0], [10, 6, 2], 1), 4)
        with self.assertRaises(ValueError):
            self.engine.icm_equity([1, -1], [1, 0])


if __name__ == "__main__":
    unittest.main()
