import itertools
import math
import random
import unittest
from icm import utility, utilities


def permutation_reference(stacks, payouts):
    alive = [i for i, s in enumerate(stacks) if s > 0]
    result = [0.] * len(stacks)
    for order in itertools.permutations(alive):
        probability, remaining = 1., sum(stacks)
        for i in order:
            probability *= stacks[i] / remaining
            remaining -= stacks[i]
        for rank, i in enumerate(order):
            if rank < len(payouts):
                result[i] += probability * payouts[rank]
    return result


class ICMTests(unittest.TestCase):
    def test_equal_stacks(self):
        for n in range(1, 9):
            p = [float(i) for i in range(n, 0, -1)]
            u = utilities([100] * n, p)
            for value in u:
                self.assertAlmostEqual(value, sum(p) / n)

    def test_winner_take_all(self):
        stacks = [10, 20, 30, 0]
        for actual, expected in zip(utilities(stacks, [6]), [1., 2., 3., 0.]):
            self.assertAlmostEqual(actual, expected)

    def test_heads_up(self):
        self.assertAlmostEqual(utility([1, 3], [100, 20], 0), 40)
        self.assertAlmostEqual(utility([1, 3], [100, 20], 1), 80)

    def test_zeros_and_awarded_tail(self):
        self.assertEqual(utilities([0, 3, 0], [50]), [0., 50., 0.])
        self.assertEqual(utilities([0, 0], []), [0., 0.])
        with self.assertRaises(ValueError):
            utilities([0, 3, 0], [50, 30, 20])

    def test_permutation_reference_and_conservation(self):
        rng = random.Random(9411)
        for n in range(2, 7):
            for _ in range(12):
                stacks = [rng.randrange(1, 1000) for i in range(n)]
                payouts = sorted([rng.randrange(0, 100) for i in range(n)], reverse=True)
                expected = permutation_reference(stacks, payouts)
                actual = utilities(stacks, payouts)
                for a, e in zip(actual, expected):
                    self.assertAlmostEqual(a, e, places=9)
                self.assertAlmostEqual(sum(actual), sum(payouts), places=9)

    def test_scale_invariance(self):
        a = utilities([1, 2, 4, 8], [100, 30, 10, 0])
        b = utilities([1000, 2000, 4000, 8000], [100, 30, 10, 0])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y)

    def test_iterables(self):
        self.assertEqual(utilities(iter([1, 3]), iter([100, 20])), [40., 80.])

    def test_invalid(self):
        for stacks, payouts in [([-1, 2], [1]), ([math.nan, 1], [1]), ([1, 2], [1, 2]),
                                 ([1, 2], [-1]), ([1, 2], [math.inf]), ([0, 0], [1])]:
            with self.assertRaises(ValueError):
                utilities(stacks, payouts)
        with self.assertRaises(ValueError):
            utility([1, 2], [1], -1)


if __name__ == '__main__':
    unittest.main()
