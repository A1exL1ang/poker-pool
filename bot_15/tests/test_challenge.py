import unittest

from benchmarks.challenge import EquityBaseline, PolarizedBettor, _forced_chop, _pot_share
from benchmarks.engine import play_table
from benchmarks.opponents import BaselineAgent
from bot.protocol import validate_observation


def cards(s):
    return ['23456789TJQKA'.index(x[0]) * 4 + 'cdhs'.index(x[1]) for x in s.split()]


class ChallengeTests(unittest.TestCase):
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            EquityBaseline('unknown')
        for params in ({'bluff_frequency': -1}, {'bluff_frequency': 2}, {'bet_fraction': 0}):
            with self.assertRaises(ValueError):
                PolarizedBettor(**params)

    def test_forced_chops(self):
        for s in ('Ac Kd Qh Js Tc', 'Ac Ad Ah As Kc', 'Kc Kd Kh Ks Ac', 'Ac Kc Qc Jc Tc'):
            self.assertTrue(_forced_chop(cards(s)))
        self.assertFalse(_forced_chop(cards('Ac Kc Qc Jd Th')))
        self.assertFalse(_forced_chop(cards('Ac Ad Ah As Qc')))

    def test_sidepot_equity_weights_small_allin_only_in_main_pot(self):
        obs = dict(hero=0, to_call=100, stacks=[900, 900, 0], contributions=[0, 100, 1],
                   in_hand=[True, True, True])
        expected, total = _pot_share(obs, .5)
        self.assertEqual(total, 201)
        self.assertEqual(expected, 3 * .25 + 198 * .5)

    def test_hero_ineligible_sidepot_excluded(self):
        obs = dict(hero=0, to_call=500, stacks=[10, 900, 900], contributions=[0, 500, 500],
                   in_hand=[True, True, True])
        expected, total = _pot_share(obs, .5)
        self.assertEqual(total, 30)
        self.assertEqual(expected, 7.5)

    def test_legal_play_and_reset_determinism(self):
        def agents(seed):
            return [EquityBaseline('tight', seed), EquityBaseline('balanced', seed + 1),
                    EquityBaseline('loose', seed + 2), PolarizedBettor(seed + 3),
                    PolarizedBettor(seed + 4, .45, 1.5), BaselineAgent('shove', seed + 5),
                    BaselineAgent('calling', seed + 6), BaselineAgent('random', seed + 7)]

        observed = [0]

        def observe(obs, action, elapsed):
            validate_observation(obs)
            self.assertIs(type(action['amount']), int)
            observed[0] += 1

        for seed in range(12):
            result = play_table(agents(seed), seed + 8181, max_hands=60, observer=observe)
            self.assertEqual(sum(result['stacks']), 16000)
            self.assertEqual(result['invalid_actions'], 0)
            self.assertEqual(result['errors'], 0)
        self.assertGreater(observed[0], 1000)
        a = agents(42)
        self.assertEqual(play_table(a, 779, max_hands=15), play_table(a, 779, max_hands=15))


if __name__ == '__main__':
    unittest.main()
