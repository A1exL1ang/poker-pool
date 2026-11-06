import copy
import random
import unittest

from agent import Agent
from bot.evaluator import card
from bot.icm import equity, call_threshold
from bot.policy import normalize
from simulator import icm_values, play_hand
from baselines import Baseline


def observation(hole=('As', 'Ah'), board=(), stacks=(1000, 1000), bets=(10, 20)):
    board = [card(c) for c in board]
    return dict(hero=0, button=0, hand_id=1, street={0: 0, 3: 1, 4: 2, 5: 3}[len(board)],
                hole=[card(c) for c in hole], board=board,
                players=[dict(stack=s, street_bet=b, invested=b, folded=False, all_in=s == 0)
                         for s, b in zip(stacks, bets)],
                pot=sum(bets), to_call=max(0, bets[1] - bets[0]), big_blind=20, small_blind=10,
                min_raise_to=max(bets) + 20, max_raise_to=bets[0] + stacks[0],
                can_raise=stacks[1] > 0 and stacks[0] > bets[1] - bets[0], history=[])


class GuardTests(unittest.TestCase):
    def test_raise_to_not_increment(self):
        o = observation()
        self.assertEqual(normalize({'action': 'raise', 'amount': 30}, o), {'action': 'raise', 'amount': 40})
        self.assertEqual(normalize({'action': 'raise', 'amount': 9000}, o), {'action': 'raise', 'amount': 1010})

    def test_short_allin_and_no_reopening(self):
        o = observation(stacks=(15, 1000))
        self.assertEqual(normalize({'action': 'raise', 'amount': 40}, o), {'action': 'raise', 'amount': 25})
        o['can_raise'] = False
        self.assertEqual(normalize({'action': 'raise', 'amount': 40}, o), {'action': 'fold'})

    def test_free_fold_and_invalid_action(self):
        o = observation(bets=(20, 20))
        for a in ({'action': 'fold'}, {'action': 'raise', 'amount': float('nan')}, None):
            self.assertEqual(normalize(a, o), {'action': 'check'})

    def test_malformed_observation_fallback_and_strict(self):
        o = observation(hole=('As', 'As'))
        a = Agent()
        self.assertEqual(a.act(o), {'action': 'fold'})
        self.assertEqual(a.errors, 1)
        self.assertEqual(a.fallbacks, 1)
        with self.assertRaises(ValueError):
            Agent(strict=True).act(o)


class ICMTests(unittest.TestCase):
    def test_independent_reference_parity(self):
        rng = random.Random(16165)
        for n in range(2, 9):
            for _ in range(8):
                stacks = [rng.randint(1, 3000) for _ in range(n)]
                payouts = sorted([rng.randint(0, 100) for _ in range(n)], reverse=True)
                reference = icm_values(stacks, payouts)
                for i in range(n):
                    self.assertAlmostEqual(equity(stacks, payouts, i), reference[i], places=9)

    def test_chip_linear_winner_take_all(self):
        self.assertAlmostEqual(equity([100, 300, 600], [1, 0, 0], 0), 0.1)

    def test_bubble_threshold_and_missing_payouts(self):
        o = observation(stacks=(500, 0, 1000), bets=(500, 1000, 0))
        o['players'][2]['folded'] = True
        self.assertIsNone(call_threshold(o, 1))
        o['payouts'] = [100, 0, 0]
        self.assertAlmostEqual(call_threshold(o, 1), 0.25)
        o['payouts'] = [50, 30, 20]
        self.assertGreater(call_threshold(o, 1), 0.25)

    def test_sidepot_guard(self):
        o = observation(stacks=(100, 0, 500), bets=(100, 1000, 400))
        o['players'][2]['folded'] = True
        o['payouts'] = [50, 30, 20]
        self.assertIsNone(call_threshold(o, 1))


class StrategyTests(unittest.TestCase):
    def test_premium_open(self):
        o = observation()
        self.assertEqual(Agent(strict=True).act(o)['action'], 'raise')

    def test_royal_board_never_folds_to_margin(self):
        o = observation(('2c', '3d'), ('As', 'Ks', 'Qs', 'Js', 'Ts'), (100000, 0), (0, 100000))
        self.assertEqual(Agent(strict=True).act(o), {'action': 'call'})

    def test_private_nuts_never_fold(self):
        o = observation(('As', 'Ks'), ('Qs', 'Js', 'Ts', '2c', '3d'), (1000, 0), (0, 1000))
        self.assertEqual(Agent(strict=True).act(o), {'action': 'call'})

    def test_locked_quads_and_broadway_chops(self):
        boards = [('2c', '2d', '2h', '2s', 'Ac'),
                  ('Ac', 'Ad', 'Ah', 'As', 'Kc'),
                  ('Ac', 'Kd', 'Qh', 'Js', 'Tc')]
        for board in boards:
            o = observation(('3c', '4d'), board, (100000, 0), (0, 100000))
            o['payouts'] = [70, 30]
            self.assertEqual(Agent(strict=True).act(o), {'action': 'call'})

    def test_reset_and_input_immutability(self):
        o = observation()
        before = copy.deepcopy(o)
        a = Agent(seed=77, strict=True)
        first = a.act(o)
        self.assertEqual(o, before)
        a.reset(77)
        self.assertEqual(a.act(o), first)
        self.assertEqual(a.errors, 0)

    def test_generated_legal_hands(self):
        for seed in range(100):
            a = Agent(seed=seed, strict=True)
            agents = [a] + [Baseline(s, seed + i) for i, s in enumerate(('tag', 'lag', 'station', 'maniac', 'random', 'nit', 'jam'))]
            r = play_hand(agents, [2000] * 8, button=seed % 8, seed=160100 + seed)
            self.assertEqual(sum(r.stacks), 16000)
            self.assertFalse([v for v in r.violations if v['seat'] == 0])
            self.assertEqual(a.errors, 0)


if __name__ == '__main__':
    unittest.main()
