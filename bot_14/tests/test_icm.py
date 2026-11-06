import itertools
import unittest
from pokerbot.icm import equity, call_threshold


def slow(stacks, hero, prizes):
    value = 0.0
    for order in itertools.permutations(range(len(stacks))):
        remaining = sum(stacks)
        p = 1.0
        for i in order:
            p *= stacks[i] / remaining
            remaining -= stacks[i]
        value += p * prizes[order.index(hero)]
    return value


class ICMTests(unittest.TestCase):
    def test_matches_permutations(self):
        for stacks in ((1, 2, 3), (1, 100, 7, 5), (2000,) * 5):
            prizes = tuple(max(0, 4 - i) for i in range(len(stacks)))
            for hero in range(len(stacks)):
                self.assertAlmostEqual(equity(stacks, hero, prizes), slow(stacks, hero, prizes))

    def test_sum_and_winner_take_all(self):
        stacks = (1, 5, 11, 30, 7, 12, 100, 29)
        prizes = (50, 30, 20, 10, 5, 2, 1, 0)
        self.assertAlmostEqual(sum(equity(stacks, h, prizes) for h in range(8)), sum(prizes))
        for h in range(8):
            self.assertAlmostEqual(equity(stacks, h, (1,)), stacks[h] / sum(stacks))

    def test_equal_payouts(self):
        self.assertEqual(equity((10, 5, 2), 0, (7, 7, 7)), 7)
        self.assertEqual(equity((0, 0, 10), 0, (5, 2, 0)), 1)

    def test_call_risk_premium(self):
        obs = dict(players=[dict(stack=100, invested=0), dict(stack=0, invested=100), dict(stack=1, invested=0)], hero=0,
                   pot=100, payouts=(.5, .3, .2))
        self.assertGreater(call_threshold(obs, 1, 100, 100), .5)
        obs.pop("payouts")
        self.assertEqual(call_threshold(obs, 1, 100, 100), .5)


if __name__ == "__main__":
    unittest.main()
