"""End-to-end policy contracts and payoff regressions, independent of tuning."""

import gzip
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from bot13.agent import Agent
from bot13.cards import card, evaluate, reference_evaluate
from bot13.icm import equity
from bot13.interface import legalize, validate
from lab.baselines import RandomLegal
from lab.simulator import Hand, Tournament


class Capture:
    def __init__(self, agent):
        self.agent = agent
        self.states = []

    def act(self, obs):
        validate(obs)
        self.states.append(obs)
        return self.agent.act(obs)


class PolicyTests(unittest.TestCase):
    def test_native_is_enabled_when_built(self):
        from bot13.native import native_available
        self.assertEqual(Agent().native is not None, native_available)

    def test_python_evaluator_agrees_with_independent_reference(self):
        rng = random.Random(130806)
        for n in (5, 6, 7):
            for _ in range(400):
                cards = rng.sample(range(52), n)
                self.assertEqual(evaluate(cards), reference_evaluate(cards))

    def test_icm_chip_share_and_prize_conservation(self):
        rng = random.Random(130807)
        for n in range(2, 9):
            stacks = [rng.randrange(1, 5000) for _ in range(n)]
            payouts = sorted([rng.randrange(100) for _ in range(n)], reverse=True)
            values = [equity(stacks, payouts, hero) for hero in range(n)]
            self.assertAlmostEqual(sum(values), sum(payouts), places=8)
            for hero in range(n):
                self.assertAlmostEqual(equity(stacks, [1] + [0] * (n - 1), hero), stacks[hero] / sum(stacks))

    def test_illegal_raise_clamped_and_short_allin_only(self):
        obs = {"to_call": 10, "can_raise": True, "min_raise_to": 100,
               "max_raise_to": 80, "current_bet": 60}
        self.assertEqual(legalize({"action": "raise", "amount": 70}, obs), {"action": "raise", "amount": 80})
        obs["can_raise"] = False
        self.assertEqual(legalize({"action": "raise", "amount": 80}, obs), {"action": "fold"})

    def test_three_way_sidepot_expected_value(self):
        hero = {"seat": 0, "committed": 20, "stack": 180, "folded": False}
        deep = {"seat": 1, "committed": 200, "stack": 0, "folded": False}
        short = {"seat": 2, "committed": 50, "stack": 0, "folded": False}
        dead = {"seat": 3, "committed": 30, "stack": 0, "folded": True}
        obs = {"players": [hero, deep, short, dead]}
        q = Agent._pot_equity(obs, hero, [deep, short], 180, (0.75, 0.25))
        # 180-chip main pot at25% plus300-chip side pot at75%.
        self.assertAlmostEqual(q, (180 * 0.25 + 300 * 0.75) / 480)

    def test_no_online_or_file_access_during_valid_actions(self):
        agent = Agent(seed=130811)
        obs = Hand([2000] * 3, [RandomLegal(i) for i in range(3)], 0, 10, 20).observation(0)
        with patch("builtins.open", side_effect=AssertionError("runtime file access")), patch("socket.socket", side_effect=AssertionError("runtime socket access")):
            action = agent.act(obs)
        self.assertIn(action["action"], ("fold", "call", "raise"))
        self.assertEqual(agent.fallbacks, 0)

    def test_full_tournaments_canonical_contract_and_legality(self):
        for native in (False, True):
            for seed in range(6):
                a = Agent(seed=130820 + seed, config={"native": native})
                capture = Capture(a)
                agents = [capture] + [RandomLegal(130840 + seed * 9 + i) for i in range(7)]
                result = Tournament(agents, blind_schedule=[(10, 20), (40, 80), (100, 200), (300, 600)],
                                    seed=130860 + seed,
                                    payouts=[50, 30, 20, 0, 0, 0, 0, 0]).run()
                self.assertEqual(sum(result.stacks), 16000)
                self.assertTrue(all(r.fallback is None for r in result.decisions if r.seat == 0))
                self.assertEqual((a.fallbacks, a.native_errors), (0, 0))
                self.assertTrue(capture.states)

    def test_reset_restores_table_state_and_seed(self):
        obs = Hand([2000] * 3, [RandomLegal(i) for i in range(3)], 0, 10, 20).observation(0)
        a, b = Agent(seed=1400), Agent(seed=1400)
        first = a.act(obs)
        a.reset(seed=1400)
        self.assertEqual(a.act(obs), first)
        self.assertEqual(b.act(obs), first)
        self.assertEqual(a.opponents[0].hands, 0)  # no prior observed action

    def test_bad_input_fallback_is_safe(self):
        a = Agent()
        self.assertEqual(a.act({"to_call": 0}), {"action": "check"})
        self.assertEqual(a.act({"to_call": 10}), {"action": "fold"})
        self.assertEqual(a.fallbacks, 2)
        obs = Hand([2000] * 3, [RandomLegal(i) for i in range(3)], 0, 10, 20).observation(0)
        obs["hero"] = 99
        self.assertEqual(a.act(obs), {"action": "fold"})
        self.assertEqual(a.fallbacks, 3)


if __name__ == "__main__":
    unittest.main()
