"""Independent ICM cross-checks and table boundary validation."""

from __future__ import annotations

import itertools
import math
import random
import unittest

from pokerbot.game import Table, icm_equities


def exhaustive_icm(stacks, payouts):
    values = [0.0] * len(stacks)
    for order in itertools.permutations(range(len(stacks))):
        total = sum(stacks)
        probability = 1.0
        for seat in order:
            probability *= stacks[seat] / total
            total -= stacks[seat]
        for place, seat in enumerate(order):
            values[seat] += probability * payouts[place]
    return values


class ICMTests(unittest.TestCase):
    def test_winner_take_all_equals_chip_fraction(self):
        values = icm_equities([100, 250, 650], [1, 0, 0])
        for actual, expected in zip(values, (0.1, 0.25, 0.65)):
            self.assertAlmostEqual(actual, expected)

    def test_heads_up_affine_payout_formula(self):
        values = icm_equities([25, 75], [0.7, 0.3])
        self.assertAlmostEqual(values[0], 0.3 + 0.4 * 0.25)
        self.assertAlmostEqual(values[1], 0.3 + 0.4 * 0.75)

    def test_equal_stacks_share_total_rewards_equally(self):
        values = icm_equities([100] * 8, [0.5, 0.25, 0.15, 0.1, 0, 0, 0, 0])
        for value in values:
            self.assertAlmostEqual(value, 0.125)

    def test_dynamic_program_matches_exhaustive_orders(self):
        rng = random.Random(137)
        for n in range(2, 7):
            for _ in range(3):
                stacks = [rng.randint(1, 4000) for _ in range(n)]
                payouts = sorted([rng.random() for _ in range(n)], reverse=True)
                with self.subTest(n=n, stacks=stacks):
                    expected = exhaustive_icm(stacks, payouts)
                    actual = icm_equities(stacks, payouts)
                    for a, e in zip(actual, expected):
                        self.assertAlmostEqual(a, e, places=10)
                    self.assertAlmostEqual(sum(actual), sum(payouts))

    def test_icm_invariant_to_chip_scale(self):
        one = icm_equities([10, 30, 100, 200], [0.5, 0.3, 0.2, 0])
        two = icm_equities([10000, 30000, 100000, 200000], [0.5, 0.3, 0.2, 0])
        for a, b in zip(one, two):
            self.assertAlmostEqual(a, b)

    def test_equal_prizes_have_no_chip_risk(self):
        values = icm_equities([1, 10, 100, 10000], [5, 5, 5, 5])
        for value in values:
            self.assertAlmostEqual(value, 5)

    def test_bad_icm_inputs_rejected(self):
        for stacks, payouts in (
            ([], []), ([0, 100], [1, 0]), ([-1, 100], [1, 0]),
            ([1, math.nan], [1, 0]), ([1, 2], [0, 1]),
            ([1, 2], [1]), ([1, 2], [math.inf, 0]),
        ):
            with self.subTest(stacks=stacks, payouts=payouts):
                with self.assertRaises(ValueError):
                    icm_equities(stacks, payouts)


class BoundariesTests(unittest.TestCase):
    def test_invalid_stacks_blinds_and_limits_rejected(self):
        for options in (
            {"stacks": [100]}, {"stacks": [100] * 9}, {"stacks": [0, 100]},
            {"stacks": [100.0, 100]}, {"stacks": [True, 100]},
            {"big_blind": 0}, {"small_blind": 21, "big_blind": 20},
            {"button": -1}, {"decision_timeout_ms": 0}, {"max_hands": 201},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    Table(**options)


if __name__ == "__main__":
    unittest.main()
