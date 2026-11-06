"""Verify the trained endgame component is applied in its declared contexts."""

import random
import unittest

from bot13.agent import Agent
from lab.baselines import RandomLegal
from lab.simulator import Hand, Tournament


class EndgamePolicyTests(unittest.TestCase):
    def test_heads_up_short_sb_aces_shoves(self):
        h = Hand([200, 200], [RandomLegal(1), RandomLegal(2)], 0, 10, 20)
        obs = h.observation(0)
        obs["hole_cards"] = [48, 51]
        a = Agent(seed=13, config={"pushfold": True})
        self.assertEqual(a.act(obs), {"action": "raise", "amount": 200})
        self.assertEqual(a.fallbacks, 0)

    def test_heads_up_short_bb_aces_calls_open_shove(self):
        h = Hand([200, 200], [RandomLegal(1), RandomLegal(2)], 0, 10, 20)
        h._apply(0, {"action": "raise", "amount": 200})
        obs = h.observation(1)
        obs["hole_cards"] = [48, 51]
        a = Agent(seed=13, config={"pushfold": True})
        self.assertEqual(a.act(obs), {"action": "call"})
        self.assertEqual(a.fallbacks, 0)

    def test_endgame_tournaments_legal_and_conserve_chips(self):
        for seed in range(16):
            agents = [Agent(seed=131900 + seed * 7 + i, config={"pushfold": True}) for i in range(2)]
            result = Tournament(agents, blind_schedule=[(10, 20), (20, 40)],
                                stacks=[150 + seed * 17, 600 - seed * 13],
                                seed=132100 + seed, hand_limit=100).run()
            self.assertEqual(sum(result.stacks), 750 + seed * 4)
            self.assertTrue(all(r.fallback is None for r in result.decisions))
            self.assertTrue(all(a.fallbacks == a.native_errors == 0 for a in agents))

    def test_reset_without_seed_restores_initial_rng(self):
        a = Agent(seed=131929)
        initial = a._random()
        for _ in range(20):
            a._random()
        a.reset()
        self.assertEqual(a._random(), initial)
        self.assertFalse(a.opponents)


if __name__ == "__main__":
    unittest.main()
