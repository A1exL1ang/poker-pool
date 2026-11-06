import itertools
import math
import unittest

from agent import Agent
from risk import icm_equity, call_threshold


def observation(hole=(48, 49), board=(), stack=1980, call=20, opponents=1):
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    players = [dict(seat=0, stack=stack, bet=0, committed=20, folded=False, all_in=False)]
    for seat in range(1, opponents + 1):
        players.append(dict(seat=seat, stack=1980-call, bet=call, committed=20+call, folded=False, all_in=False))
    return dict(hand_id=1, hero=0, button=0, street=street, hole=list(hole), board=list(board),
                players=players, pot=sum(p['committed'] for p in players), big_blind=20, small_blind=10,
                to_call=min(stack, call), min_raise_to=max(20, call*2), max_raise_to=stack,
                can_raise=True, legal_actions=['fold', 'call', 'raise'] if call else ['fold','check','raise'], history=[])


class ICMTests(unittest.TestCase):
    def test_equal_and_winner_take_all(self):
        self.assertAlmostEqual(icm_equity([100]*8, [1,.7,.4,.2,.1,0,0,0]), .3)
        self.assertAlmostEqual(icm_equity([300,100], [100,0]), 75)
        self.assertAlmostEqual(icm_equity([300,100], [100,100]), 100)

    def test_against_permutations(self):
        stacks = [23, 51, 79, 102, 130]
        pay = [15, 8, 5, 1, 0]
        reference = [0.] * 5
        for order in itertools.permutations(range(5)):
            prob = 1.
            total = sum(stacks)
            for i in order:
                prob *= stacks[i] / total
                total -= stacks[i]
            for place, i in enumerate(order):
                reference[i] += prob * pay[place]
        for i in range(5):
            self.assertAlmostEqual(icm_equity(stacks, pay, i), reference[i])

    def test_call_threshold_chip_utility(self):
        players = [dict(seat=0, stack=100, committed=0), dict(seat=1, stack=0, committed=100)]
        self.assertAlmostEqual(call_threshold(players,0,1,100,100,[1,0]), .5)

    def test_bubble_risk(self):
        p = [dict(seat=0, stack=90, committed=10), dict(seat=1, stack=0, committed=100), dict(seat=2, stack=30, committed=0)]
        threshold = call_threshold(p,0,1,90,110,[.5,.3,.2])
        self.assertGreater(threshold, .45)
        self.assertLess(threshold, .9)


class AgentTests(unittest.TestCase):
    def test_aces_open(self):
        o = observation()
        self.assertEqual(Agent().act(o)['action'], 'raise')

    def test_check_safe_failure(self):
        o = observation(board=(0, 4, 8), call=0)
        o['hole'] = [0,0]
        agent = Agent()
        self.assertEqual(agent.act(o), {'action':'check'})
        self.assertEqual(agent.fallbacks, 1)

    def test_fold_safe_failure(self):
        o = observation()
        o['hole'] = [90, 2]
        self.assertEqual(Agent().act(o), {'action':'fold'})

    def test_short_allin_raise_to(self):
        o = observation(stack=30, call=20)
        self.assertEqual(Agent().act(o), {'action':'raise', 'amount':30})

    def test_no_reopening(self):
        o = observation()
        o['can_raise'] = False
        o['legal_actions'].remove('raise')
        self.assertEqual(Agent().act(o)['action'], 'call')

    def test_shared_royal_does_not_fold(self):
        o = observation(hole=(0,5), board=(32,36,40,44,48), call=20)
        o['players'][1]['all_in'] = True
        o['players'][1]['stack'] = 0
        self.assertEqual(Agent().act(o)['action'], 'call')

    def test_eligible_pot_excludes_unwinnable_sidepot(self):
        o = observation(stack=10, call=100, opponents=2)
        self.assertEqual(Agent._eligible(o, o['players'][0]), 80)

    def test_reset(self):
        a = Agent()
        a.stats[1] = [100]*6
        a.reset()
        self.assertEqual(a.stats, {})
        self.assertEqual(a.decisions, 0)

    def test_scalar_malformed_observation(self):
        for o in (None, [], 1, 'bad'):
            self.assertEqual(Agent().act(o), {'action':'fold'})

    def test_seed_persists_reset(self):
        a = Agent(123)
        expected = a.rng.random()
        a.reset()
        self.assertEqual(a.rng.random(), expected)

    def test_shared_aces_king_board(self):
        o = observation(hole=(0,5), board=(48,49,50,51,44), stack=200, call=200)
        o['players'][1]['stack'] = 0
        o['players'][1]['all_in'] = True
        o['payouts'] = [.5,.3,.2,0,0,0,0,0]
        for seat in range(2,8):
            o['players'].append(dict(seat=seat,stack=1000,bet=0,committed=0,folded=True,all_in=False))
        self.assertEqual(Agent().act(o)['action'], 'call')

    def test_shared_rainbow_broadway(self):
        o = observation(hole=(0,5), board=(48,45,42,39,32), stack=200, call=200)
        o['players'][1]['stack'] = 0
        o['players'][1]['all_in'] = True
        o['payouts'] = [.5,.3,.2,0,0,0,0,0]
        for seat in range(2,8):
            o['players'].append(dict(seat=seat,stack=1000,bet=0,committed=0,folded=True,all_in=False))
        self.assertEqual(Agent().act(o)['action'], 'call')

    def test_sidepot_masks(self):
        o = observation(hole=(28,29), board=(48,45,42,23,0), stack=3000, call=1000, opponents=2)
        hero = o['players'][0]
        hero.update(bet=0,committed=1000)
        o['players'][1].update(stack=0,bet=0,committed=10,all_in=True)
        o['players'][2].update(stack=2000,bet=1000,committed=2000)
        o['pot'] = 3010
        pots = Agent._pot_layers(o,hero,o['players'][1:])
        self.assertEqual(pots, [(30,3),(3980,2)])
        o['history']=[dict(seat=2,street=3,action='raise',amount=1000,paid=1000)]
        self.assertEqual(Agent(170019).act(o)['action'], 'call')

    def test_dead_small_blind_position(self):
        from unittest.mock import patch
        from simulator import Hand
        # Seat0 dead button; seat1 dead SB; BB2 and UTG3. UTG must not
        # inherit the 42%-wide BB isolation range.
        h=Hand([0,0,2000,2000,2000,2000,2000,2000],button=0,sb_seat=1,bb_seat=2)
        o=h.observation()
        self.assertEqual(o['hero'],3)
        with patch('agent.percentile',return_value=.30):
            self.assertEqual(Agent().act(o)['action'],'fold')

    def test_decision_is_offline_and_has_no_file_reads(self):
        from unittest.mock import patch
        import socket
        a=Agent()
        o=observation(hole=(48,49),board=(4,9,14),call=20)
        with patch('builtins.open',side_effect=AssertionError('decision opened file')), \
             patch('io.open',side_effect=AssertionError('decision opened file')), \
             patch.object(socket,'socket',side_effect=AssertionError('decision opened socket')):
            result=a.act(o)
        self.assertIn(result['action'],o['legal_actions'])
        self.assertEqual(a.fallbacks,0)

    def test_randomized_legal_actions(self):
        import random
        rng = random.Random(7111)
        a = Agent()
        for i in range(600):
            cards = rng.sample(range(52), 7)
            board = cards[2:2+rng.choice([0,3,4,5])]
            stack = rng.randrange(1, 2001)
            call = rng.choice([0, 10, 20, 50, 200, 2000])
            o = observation(cards[:2], board, stack, call, rng.randrange(1,8))
            o['hand_id'] = i
            result = a.act(o)
            self.assertIn(result['action'], o['legal_actions'])
            if result['action'] == 'raise':
                self.assertLessEqual(result['amount'], o['max_raise_to'])
                self.assertTrue(result['amount'] >= o['min_raise_to'] or result['amount'] == o['max_raise_to'])
        self.assertEqual(a.fallbacks, 0)


if __name__ == '__main__':
    unittest.main()
