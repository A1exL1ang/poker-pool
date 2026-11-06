import random
import unittest

from policy.icm import value, call_threshold, shove_gain


def recursive_reference(stacks, hero, payouts):
    if len(stacks) == 1:
        return payouts[0]
    total = sum(stacks)
    out = stacks[hero] / total * payouts[0]
    for j, s in enumerate(stacks):
        if j != hero:
            rest = stacks[:j] + stacks[j + 1:]
            out += s / total * recursive_reference(rest, hero - (j < hero), payouts[1:])
    return out


class ICMTests(unittest.TestCase):
    def test_reused_texture_matches_reference(self):
        from policy.texture import texture,texture_from_features
        from equity.model import describe
        rng=random.Random(181918)
        for i in range(3000):
            cards=rng.sample(range(52),5+i%3)
            self.assertEqual(texture(cards[:2],cards[2:]),texture_from_features(describe(cards[:2],cards[2:])))

    def test_equal_stacks(self):
        for n in range(2, 9):
            pay = [float(n - i) ** 2 for i in range(n)]
            self.assertAlmostEqual(value([100] * n, 0, pay), sum(pay) / n)

    def test_exact_reference_and_sum(self):
        rng = random.Random(181)
        for n in range(2, 7):
            for _ in range(4):
                stacks = [rng.randrange(1, 10000) for _ in range(n)]
                pay = sorted([rng.randrange(100) for _ in range(n)], reverse=True)
                ev = [value(stacks, h, pay) for h in range(n)]
                for h in range(n):
                    self.assertAlmostEqual(ev[h], recursive_reference(stacks, h, pay))
                self.assertAlmostEqual(sum(ev), sum(pay))

    def test_zero_stacks_and_winner_takes_all(self):
        self.assertEqual(value([100, 0, 0], 0, [50, 30, 20]), 50)
        self.assertEqual(value([100, 0, 0], 1, [50, 30, 20]), 20)
        self.assertAlmostEqual(value([200, 300, 500], 0, [1, 0, 0]), .2)

    def test_icm_call_chip_ev_special_case(self):
        p = [{'seat': 0, 'stack': 900, 'invested': 100},
             {'seat': 1, 'stack': 700, 'invested': 300},
             {'seat': 2, 'stack': 1000, 'invested': 0}]
        self.assertAlmostEqual(call_threshold(p, 0, 1, 400, 200, [1, 0, 0]), 1 / 3)
        self.assertGreater(call_threshold(p, 0, 1, 400, 200, [50, 30, 20]), 1 / 3)

    def test_shove_chip_ev_and_constant_payout(self):
        p = [{'seat':0,'stack':900,'bet':0,'invested':0},
             {'seat':1,'stack':800,'bet':100,'invested':100},
             {'seat':2,'stack':1000,'bet':0,'invested':0}]
        # One possible caller: hero risks900 and villain adds800. Pot100.
        gain, steal = shove_gain(p,0,100,[1,0,0],.6,[(1,.4,.55)])
        chip_ev = .6*100 + .4*(.55*900 - .45*900)
        self.assertAlmostEqual(gain,chip_ev/2800)
        self.assertAlmostEqual(steal,100/2800)
        gain, steal = shove_gain(p,0,100,[10,10,10],.6,[(1,.4,.55)])
        self.assertAlmostEqual(gain,0)
        self.assertAlmostEqual(steal,0)

    def test_certain_win_never_folds_at_high_icm_threshold(self):
        from unittest.mock import patch
        from agent import Agent
        o = {'hero':0,'hand_id':0,'street':'river','button':1,'big_blind':20,
             'hole':[48,44],'board':[40,36,32,5,10],
             'players':[{'seat':0,'stack':100,'bet':0,'invested':10,'folded':False},
                        {'seat':1,'stack':0,'bet':100,'invested':110,'folded':False}],
             'history':[],'to_call':100,'pot':120,'can_raise':False,
             'payouts':[1,1],'min_raise_to':200,'max_raise_to':100}
        with patch('agent.call_threshold',return_value=.999):
            self.assertEqual(Agent(seed=18).act(o)['action'],'call')


if __name__ == '__main__':
    unittest.main()
