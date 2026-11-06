import itertools
import random
import unittest

from icm import hero_equity, call_threshold
from simulator import icm_values


class ICMTests(unittest.TestCase):
    def test_independent_full_order_enumeration(self):
        stacks = (10, 20, 30, 50, 90)
        payouts = (.45, .25, .16, .09, .05)
        expected = [0.] * len(stacks)
        for order in itertools.permutations(range(len(stacks))):
            prob, left = 1., sum(stacks)
            for i in order:
                prob *= stacks[i] / left
                left -= stacks[i]
            for place, i in enumerate(order):
                expected[i] += prob * payouts[place]
        for i in range(len(stacks)):
            self.assertAlmostEqual(hero_equity(stacks, payouts, i), expected[i], places=12)

    def test_independent_subset_implementation_and_scale(self):
        rng = random.Random(120120)
        for n in range(2, 9):
            for _ in range(12):
                stacks = [rng.randrange(1, 10000) for _ in range(n)]
                prizes = (.5, .3, .2)[:n]
                independent = icm_values(stacks, prizes)
                for i in range(n):
                    self.assertAlmostEqual(hero_equity(stacks, prizes, i), independent[i], places=11)
                    self.assertAlmostEqual(hero_equity([s*7 for s in stacks], prizes, i), independent[i], places=11)

    def test_eliminated_hero_gets_correct_place(self):
        self.assertEqual(hero_equity([0]+[100]*7, (.5,.3,.2), 0), 0)
        self.assertEqual(hero_equity([0,100,100], (.5,.3,.2), 0), .2)
        self.assertEqual(hero_equity([100,0,0], (.5,.3,.2), 0), .5)

    def test_winner_take_all_chip_odds(self):
        obs = dict(hero=0,stacks=[900,0,500],pot=1100,payouts=(1,0,0))
        self.assertAlmostEqual(call_threshold(obs,1,900,1100), 900/2000)

    def test_bubble_call_requires_more_equity(self):
        obs = dict(hero=0,stacks=[900,0,20,1800],pot=1100,payouts=(.5,.3,.2,0))
        q = call_threshold(obs,1,900,1100)
        self.assertGreater(q, .55)
        self.assertLess(q, .9)

    def test_heads_up_linear_payout_increment(self):
        obs = dict(hero=0,stacks=[900,0],pot=1100,payouts=(.7,.3))
        self.assertAlmostEqual(call_threshold(obs,1,900,1100), .45)

    def test_linear_payout_exponential_race_identity(self):
        stacks = (113, 229, 601, 990, 44, 93, 830, 19)
        prizes = tuple((8-k)/36 for k in range(8))
        independently_computed = icm_values(stacks, prizes)
        for hero in range(8):
            self.assertAlmostEqual(hero_equity(stacks, prizes, hero), independently_computed[hero], places=12)


if __name__ == '__main__':
    unittest.main()
