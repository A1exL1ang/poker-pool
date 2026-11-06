"""Exactness and reproducibility tests for generated single-player ICM."""
from pathlib import Path
import random
import unittest

from policy.icm import value
from policy.icm_fast import fast_value, fast_value_dp
from sim.icm import icm
from training.compile_icm import render


def wrapped(stacks, hero, payouts, kernel=fast_value):
    if stacks[hero] <= 0:
        return float(payouts[len(stacks) - 1])
    others = tuple(s for i, s in enumerate(stacks) if i != hero and s > 0)
    if not others:
        return float(payouts[0])
    base = payouts[len(others)]
    differences = tuple(p - base for p in payouts[:len(others)])
    last = max((i for i, p in enumerate(differences) if p), default=-1)
    return base + kernel(stacks[hero], others, differences, last)


class FastICMTests(unittest.TestCase):
    def test_generated_source_reproducible(self):
        path = Path(__file__).resolve().parent.parent / "policy" / "icm_fast.py"
        self.assertEqual(path.read_text(encoding="utf-8"), render())

    def test_all_kernels_randomized(self):
        rng = random.Random(180073)
        for m in range(1, 8):
            for last in range(m):
                for _ in range(40):
                    stacks = [rng.randrange(1, 16001) for _ in range(m + 1)]
                    base = rng.random() * 10
                    differences = sorted([rng.random() * 100 for _ in range(last + 1)], reverse=True)
                    payouts = [base + d for d in differences] + [base] * (m - last)
                    expected = value(stacks, 0, payouts)
                    actual = wrapped(stacks, 0, payouts)
                    dp = wrapped(stacks, 0, payouts, fast_value_dp)
                    self.assertAlmostEqual(actual, expected, delta=2e-10)
                    self.assertAlmostEqual(dp, expected, delta=2e-10)

    def test_arbitrary_payout_profiles_against_independent_icm(self):
        rng = random.Random(180074)
        for n in range(2, 9):
            for _ in range(15):
                stacks = [rng.randrange(1, 16001) for _ in range(n)]
                profiles = ([1] + [0] * (n - 1), [10] * n,
                            [50, 30, 20][:n] + [0] * max(0, n - 3),
                            [25] * min(n, 4) + [0] * max(0, n - 4),
                            sorted([rng.random() * 100 for _ in range(n)], reverse=True))
                for payouts in profiles:
                    expected = icm(stacks, payouts)
                    got = [wrapped(stacks, h, payouts) for h in range(n)]
                    for actual, reference in zip(got, expected):
                        self.assertAlmostEqual(actual, reference, delta=2e-10)
                    self.assertAlmostEqual(sum(got), sum(payouts), delta=2e-9)

    def test_extreme_integer_chip_ratios(self):
        profiles = ([100, 70, 50, 35, 24, 15, 8, 0], [25] * 4 + [0] * 4,
                    [50, 30, 20, 0, 0, 0, 0, 0])
        for big in (16000, 10**6, 10**12):
            for stacks in ([big] + [1] * 7, [1] + [big] * 7,
                           [1, big, 2, 3, 11, 1, 2, 3]):
                for payouts in profiles:
                    for h in (0, 1, 7):
                        # Generated DP uses direct remaining-stack sums, avoiding
                        # cancellation in the original total-minus-subset sums.
                        expected = wrapped(stacks, h, payouts, fast_value_dp)
                        actual = wrapped(stacks, h, payouts)
                        self.assertAlmostEqual(actual, expected, delta=2e-10)
                        self.assertGreaterEqual(actual, -2e-10)
                        self.assertLessEqual(actual, payouts[0] + 2e-10)

    def test_eliminated_and_equal_payout_cases(self):
        for stacks in ([100, 0, 0], [100, 200, 0], [0, 100, 200]):
            for h in range(3):
                self.assertAlmostEqual(wrapped(stacks, h, [50, 30, 20]), value(stacks, h, [50, 30, 20]))
        self.assertEqual(fast_value(100, (), (), -1), 0)
        self.assertEqual(fast_value(100, (50, 100), (0, 0), -1), 0)


if __name__ == "__main__":
    unittest.main()
