import itertools
import copy
import random
import unittest

from bot.agent import Agent
from bot.icm import equities, hero_equity
from bot.protocol import legalize, validate_observation


def observation(**changes):
    o = dict(hero=0, button=0, hand_id=1, street=0, hole=[48, 49], board=[],
             stacks=[1990, 1980], in_hand=[True, True], street_bets=[10, 20],
             contributions=[10, 20], pot=30, small_blind=10, big_blind=20,
             to_call=10, min_raise_to=40, max_raise_to=2000, can_raise=True,
             position=1, history=[])
    o.update(changes)
    return o


class ICMTests(unittest.TestCase):
    def test_permutation_reference(self):
        rng = random.Random(151)
        for n in range(2, 7):
            stacks = [rng.randrange(1, 100) for _ in range(n)]
            payouts = sorted([rng.randrange(0, 100) for _ in range(n)], reverse=True)
            expected = [0.] * n
            for order in itertools.permutations(range(n)):
                p, total = 1., sum(stacks)
                for i in order:
                    p *= stacks[i] / total
                    total -= stacks[i]
                for k, i in enumerate(order):
                    expected[i] += p * payouts[k]
            actual = equities(stacks, payouts)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b, places=9)
            self.assertAlmostEqual(sum(actual), sum(payouts), places=9)

    def test_zero_stack_tie(self):
        self.assertEqual(equities([10, 0, 0], [50, 30, 20]), [50, 25, 25])

    def test_chip_linear_and_equal(self):
        self.assertAlmostEqual(hero_equity([1, 2, 3, 4], [100, 0, 0, 0], 1), 20)
        self.assertAlmostEqual(hero_equity([1, 2, 3, 4], [5, 5, 5, 5], 1), 5)


class AgentTests(unittest.TestCase):
    def test_valid_interface(self):
        self.assertTrue(validate_observation(observation()))

    def test_raise_to_and_short_allin(self):
        o = observation()
        self.assertEqual(legalize(o, {'action': 'raise', 'amount': 50}), {'action': 'raise', 'amount': 50})
        o.update(stacks=[19, 1980], max_raise_to=29)
        self.assertEqual(legalize(o, {'action': 'raise', 'amount': 50}), {'action': 'raise', 'amount': 29})

    def test_closed_action_becomes_call(self):
        o = observation(can_raise=False)
        self.assertEqual(legalize(o, {'action': 'raise', 'amount': 500}), {'action': 'call', 'amount': 10})

    def test_allin_call_capped(self):
        o = observation(stacks=[5, 1980], max_raise_to=15)
        self.assertEqual(legalize(o, {'action': 'call', 'amount': 1000}), {'action': 'call', 'amount': 5})

    def test_free_fold_becomes_check(self):
        self.assertEqual(legalize(observation(to_call=0), {'action': 'fold'}), {'action': 'check', 'amount': 0})

    def test_premium_opens(self):
        self.assertEqual(Agent().act(observation())['action'], 'raise')

    def test_royal_board_does_not_fold(self):
        o = observation(street=3, hole=[0, 5], board=[32, 36, 40, 44, 48],
                        stacks=[100, 0, 0], street_bets=[0, 100, 100],
                        in_hand=[True] * 3, contributions=[100, 200, 200],
                        pot=500, to_call=100, min_raise_to=200, max_raise_to=100,
                        can_raise=False)
        self.assertEqual(Agent().act(o)['action'], 'call')

    def test_no_duplicate_history_updates(self):
        a = Agent()
        o = observation(history=[dict(seat=1, street=0, action='raise', amount=60)])
        a.act(o)
        saved = [s[:] for s in a.stats]
        a.act(o)
        self.assertEqual(a.stats, saved)
        a.reset()
        self.assertEqual(sum(sum(s) for s in a.stats), 0)

    def test_forced_chops_beyond_royal(self):
        for board in ([32, 37, 42, 47, 48], [0, 1, 2, 3, 48], [48, 49, 50, 51, 44]):
            o = observation(street=3, hole=[8, 13], board=board,
                            stacks=[100, 0, 0], street_bets=[0, 100, 100],
                            in_hand=[True] * 3, contributions=[100, 200, 200],
                            pot=500, to_call=100, min_raise_to=200, max_raise_to=100,
                            can_raise=False)
            self.assertEqual(Agent().act(o)['action'], 'call')

    def test_blinds_allin_no_raise_history(self):
        o = observation(hero=0, button=0, stacks=[2000, 0, 0], hole=[48, 49],
                        street_bets=[0, 10, 20], contributions=[0, 10, 20],
                        in_hand=[True] * 3, pot=30, to_call=20, min_raise_to=40,
                        max_raise_to=2000, can_raise=False, position=0)
        self.assertEqual(Agent().act(o)['action'], 'call')

    def test_tiny_allin_not_counted_in_large_sidepot(self):
        o = observation(stacks=[100, 100, 0], in_hand=[True] * 3,
                        contributions=[1000, 1000, 1], street_bets=[0, 0, 0])
        # Only3chips are three-way;1998chips are heads-up.
        actual = Agent._pot_equity(o, 0, .5, .5, 1)
        expected = (1998 * .5 + 3 * .5 * .5 ** .85) / 2001
        self.assertAlmostEqual(actual, expected)
        self.assertGreater(actual, .499)

    def test_icm_premium_on_bubble(self):
        o = observation(stacks=[1000, 0, 200, 200], in_hand=[True, True, False, False],
                        contributions=[0, 1000, 0, 0], street_bets=[0, 1000, 0, 0],
                        pot=1000, to_call=1000, payouts=[50, 30, 20, 0],
                        max_raise_to=1000, min_raise_to=2000, can_raise=False)
        p = Agent()._risk_threshold(o, 1000, 1000, 1, .5)
        self.assertGreater(p, .6)

    def test_local_smoke_and_no_internal_fallback(self):
        from benchmarks.engine import play_table
        from benchmarks.opponents import BaselineAgent
        for seed in range(5):
            a = Agent(seed)
            result = play_table([a] + [BaselineAgent(s, seed) for s in ['balanced', 'tight', 'calling', 'aggressive', 'random', 'shove', 'balanced']], seed, max_hands=50)
            self.assertEqual(result['invalid_actions'], 0)
            self.assertEqual(result['errors'], 0)
            self.assertEqual(a.errors, 0)
            self.assertEqual(sum(result['stacks']), 16000)

    def test_suit_and_seat_relabeling_do_not_change_decisions(self):
        from benchmarks.engine import play_table
        from benchmarks.opponents import BaselineAgent
        corpus = []
        def observe(o, action, elapsed):
            if len(corpus) < 100 and (o['street'] or len(corpus) < 20):
                corpus.append(o)
        play_table([BaselineAgent('balanced', i) for i in range(8)], 615, max_hands=30, observer=observe)
        for i, o in enumerate(corpus):
            expected = Agent(i).act(o)
            rotated = copy.deepcopy(o)
            n = len(o['stacks'])
            rotated['hero'] = (o['hero'] + 3) % n
            rotated['button'] = (o['button'] + 3) % n
            for key in ('stacks', 'in_hand', 'street_bets', 'contributions'):
                rotated[key] = o[key][-3:] + o[key][:-3]
            for row in rotated['history']:
                row['seat'] = (row['seat'] + 3) % n
            self.assertEqual(expected, Agent(i).act(rotated))
            for permutation in ((3, 2, 1, 0), (1, 2, 3, 0)):
                changed = copy.deepcopy(o)
                for key in ('hole', 'board'):
                    changed[key] = [4 * (c // 4) + permutation[c % 4] for c in o[key]]
                self.assertEqual(expected, Agent(i).act(changed))


if __name__ == '__main__':
    unittest.main()
