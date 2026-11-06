import unittest
from equity import (PREFLOP, PERCENTILE, RANGE_FRACTIONS, hand_type,
                    preflop_equity, preflop_percentile, equity_vs_range)


class EquityTests(unittest.TestCase):
    def test_table_complete(self):
        self.assertEqual(len(PREFLOP), 169)
        self.assertTrue(all(0 < x < 1 for row in PREFLOP for x in row))
        self.assertEqual(len(set(hand_type((a, b)) for a in range(52) for b in range(a + 1, 52))), 169)

    def test_landmarks(self):
        self.assertAlmostEqual(preflop_equity([48, 49]), .852, delta=.006)
        self.assertAlmostEqual(preflop_equity([24, 1]), .345, delta=.04)
        self.assertLess(preflop_percentile([48, 49]), preflop_percentile([44, 45]))
        self.assertLess(preflop_percentile([44, 45]), preflop_percentile([48, 44]))

    def test_interpolation_and_invariance(self):
        for a in range(52):
            for b in range(a + 1, 52):
                self.assertEqual(hand_type((a, b)), hand_type((b, a)))
                self.assertEqual(preflop_equity([a, b]), preflop_equity([a ^ 3, b ^ 3]))
                for i, fraction in enumerate(RANGE_FRACTIONS):
                    self.assertAlmostEqual(equity_vs_range((a, b), fraction), PREFLOP[hand_type((a, b))][i])
                self.assertTrue(0 < preflop_percentile((a, b)) < 1)


if __name__ == '__main__':
    unittest.main()
