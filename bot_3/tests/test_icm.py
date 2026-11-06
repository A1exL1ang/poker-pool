import unittest
from bot3.icm import equity, call_threshold


class ICMTests(unittest.TestCase):
    def test_winner_take_all_equals_chip_share(self):
        stacks = [100, 400, 700, 900, 200, 300, 500, 150]
        for i, stack in enumerate(stacks):
            self.assertAlmostEqual(equity(stacks, [1], i), stack / sum(stacks))

    def test_symmetry_and_sum(self):
        for n in range(2, 9):
            payouts = [8, 4, 2][:n]
            values = [equity([100] * n, payouts, i) for i in range(n)]
            self.assertAlmostEqual(sum(values), sum(payouts))
            self.assertAlmostEqual(max(values), min(values))
        stacks = [12, 43, 18, 52, 100]
        self.assertAlmostEqual(sum(equity(stacks, [10, 5, 2], i) for i in range(5)), 17)

    def test_chip_ev_call_threshold(self):
        self.assertAlmostEqual(call_threshold([200, 100], [1], 0, 1, 100, 150), .4)

    def test_bubble_risk_premium(self):
        self.assertGreater(call_threshold([100, 100, 5], [5, 3], 0, 1, 100, 120), 100 / 220)

    def test_eliminated_player(self):
        self.assertEqual(equity([0, 100, 200], [5, 3, 1], 0), 1)
