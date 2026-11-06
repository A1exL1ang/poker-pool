"""Independent strategy/risk regressions assembled during helper review."""
import random
import unittest

from agent import Agent
from risk import call_threshold, icm_equity
from simulator import icm


class RiskAuditTests(unittest.TestCase):
    def test_icm_against_independent_subset_model_randomized(self):
        rng = random.Random(170020)
        for _ in range(500):
            n = rng.randrange(2, 9)
            stacks = [rng.randrange(0, 3000) if rng.random() > .15 else 0 for _ in range(n)]
            hero = rng.randrange(n)
            stacks[hero] = max(1, stacks[hero])
            payouts = sorted([rng.randrange(0, 100) for _ in range(n)], reverse=True)
            expected = icm(stacks, payouts)[hero]
            self.assertAlmostEqual(icm_equity(stacks, payouts, hero), expected, places=9)

    def test_icm_elimination_pays_next_occupied_place(self):
        # The helper is called with the funded-at-hand-start population;
        # an individual losing all-in occupies its final place in that list.
        self.assertEqual(icm_equity([0, 1000, 2000], [50, 30, 20]), 20)

    def test_call_threshold_matches_independent_settled_states(self):
        players = [dict(seat=0, stack=100, committed=100),
                   dict(seat=1, stack=100, committed=200)]
        players += [dict(seat=i, stack=1000, committed=10) for i in range(2, 8)]
        payouts = [50, 30, 20, 0, 0, 0, 0, 0]
        fold = icm([100, 460] + [1000] * 6, payouts)[0]
        win = icm([460, 100] + [1000] * 6, payouts)[0]
        lose = icm([0, 560] + [1000] * 6, payouts)[0]
        expected = (fold - lose) / (win - lose)
        actual = call_threshold(players, 0, 1, 100, 360, payouts)
        self.assertAlmostEqual(actual, expected)

    def test_winner_take_all_call_threshold_is_chip_pot_odds(self):
        rng = random.Random(170021)
        for _ in range(100):
            call = rng.randrange(1, 2000)
            hero_committed = rng.randrange(0, 2000)
            excess = rng.randrange(0, 2000)
            hero_stack = call
            villain_committed = hero_committed + call + excess
            players = [dict(seat=0, stack=hero_stack, committed=hero_committed),
                       dict(seat=1, stack=rng.randrange(0, 2000), committed=villain_committed)]
            players += [dict(seat=i, stack=rng.randrange(1, 2000), committed=0) for i in range(2, 8)]
            eligible = hero_committed * 2 + call
            expected = call / (eligible + call)
            actual = call_threshold(players, 0, 1, call, eligible, [1, 0, 0, 0, 0, 0, 0, 0])
            self.assertAlmostEqual(actual, expected)


class InputAuditTests(unittest.TestCase):
    def test_non_mapping_observations_fold(self):
        agent = Agent()
        for value in (None, [], 1, 2.5, "invalid"):
            self.assertEqual(agent.act(value), {"action": "fold"})


if __name__ == "__main__":
    unittest.main()
