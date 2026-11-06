import random
import unittest

from baselines import BASELINES, BASE_REFERENCE_PARAMS, MonteCarloAgent, make_baseline
from engine import Hand
from evaluator import evaluate


class BaselineTests(unittest.TestCase):
    def test_all_styles_return_legal_actions(self):
        for style in BASELINES:
            agents = [MonteCarloAgent(seed=i, samples=4) if style == 'mc' else make_baseline(style, seed=i)
                      for i in range(3)]
            h = Hand([100, 50, 200], 0, 5, 10, rng=random.Random(14), evaluator=evaluate)
            h.play(agents)
            self.assertEqual(h.errors, 0, style)
            self.assertEqual(sum(h.stacks), 350)

    def test_mc_deterministic_and_observation_unchanged(self):
        h = Hand([100, 100], 0, 5, 10, rng=random.Random(31), evaluator=evaluate)
        obs = h.observation()
        a, b = MonteCarloAgent(seed=31, samples=16), MonteCarloAgent(seed=31, samples=16)
        before = repr(obs)
        self.assertEqual(a.estimate(obs), b.estimate(obs))
        self.assertEqual(before, repr(obs))

    def test_mc_contestable_sidepots_and_board_tie(self):
        obs = dict(hero=0, hole=[0, 4], board=[32, 36, 40, 44, 48],
                   stacks=[100, 0, 0], folded=[False]*3, contributions=[10, 10, 100],
                   to_call=0, history=[])
        # Royal flush on board ties the 30-chip hero-contestable pot. Hero
        # cannot win any of the 90-chip side layer funded only by seat2.
        equity, award, possible = MonteCarloAgent(seed=4, samples=12).estimate(obs)
        self.assertAlmostEqual(equity, 1/3)
        self.assertAlmostEqual(award, 10.)
        self.assertEqual(possible, 30)

    def test_reference_params_are_frozen_and_table_state_is_private(self):
        from bot import DEFAULTS
        original_defaults = dict(DEFAULTS)
        try:
            DEFAULTS['open_scale'] = 9.0
            DEFAULTS['future_test_key'] = 4.0
            a = make_baseline('reference', seed=31)
            b = make_baseline('reference', seed=31)
            self.assertEqual(a.params, BASE_REFERENCE_PARAMS)
            self.assertEqual(b.params, BASE_REFERENCE_PARAMS)
            self.assertIsNot(a.params, b.params)
            self.assertIsNot(a.stats, b.stats)
            self.assertIsNot(a._seen_pre, b._seen_pre)
            state = b.state
            a._random()
            a.errors = 10
            a.params['open_scale'] = 4.0
            a._seen_pre.add(2)
            a.stats.append({'test': 1})
            self.assertEqual(b.state, state)
            self.assertEqual(b.errors, 0)
            self.assertEqual(b.params, BASE_REFERENCE_PARAMS)
            self.assertEqual(b._seen_pre, set())
            self.assertEqual(b.stats, [])
        finally:
            DEFAULTS.clear()
            DEFAULTS.update(original_defaults)


if __name__ == '__main__':
    unittest.main()
