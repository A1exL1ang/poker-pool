"""Independent policy audit regressions; local mechanics and numerical checks."""
from __future__ import annotations
import gzip
import json
from pathlib import Path
import random
import unittest

from bot13.agent import Agent
from bot13.cards import evaluate, features
from bot13.icm import equity
from bot13.interface import validate
from bot13.native import NativeEvaluator, native_available

ROOT = Path(__file__).resolve().parents[1]


class PolicyAuditTests(unittest.TestCase):
    def test_no_draw_feature_for_already_made_straight(self):
        self.assertEqual(evaluate([48, 49, 0, 5, 10, 15]) >> 24, 4)
        f = features([48, 49], [0, 5, 10, 15])
        self.assertFalse(f["straight_draw"])
        self.assertFalse(f["open_ended"])

    def test_no_private_draw_when_best_straight_plays_board(self):
        self.assertFalse(features([0, 5], [16, 21, 26, 31])["open_ended"])
        self.assertFalse(features([0, 45], [8, 13, 18, 23])["straight_draw"])

    def test_private_open_ended_draw_remains_detected(self):
        f = features([16, 21], [8, 13, 48])  # 65 on 43A: 2 or 7 makes a private straight.
        self.assertTrue(f["straight_draw"])
        self.assertTrue(f["open_ended"])

    def test_winner_take_all_small_pot_has_zero_icm_risk_premium(self):
        players = [dict(seat=i, stack=1000, committed=100, bet=100, folded=False, all_in=False) for i in range(8)]
        observation = {"players": players, "hero": 0, "payouts": [1, 0, 0, 0, 0, 0, 0, 0]}
        agent = Agent(config={"native": False})
        self.assertEqual(agent._risk_premium(observation, players[0], players[1:], 100, 800), 0)

    def test_allin_sidepot_layers_and_folded_dead_chips(self):
        hero = dict(seat=0, stack=100, committed=100)
        deep = dict(seat=1, stack=0, committed=300)
        short = dict(seat=2, stack=0, committed=50)
        dead = dict(seat=3, stack=100, committed=150, folded=True)
        observation = {"players": [hero, deep, short, dead]}
        # Main 50*4 vs two opponents; remaining400 vs deep opponent alone.
        q = Agent._pot_equity(observation, hero, [deep, short], 100, (.8, .25))
        self.assertAlmostEqual(q, (200*.25+400*.8)/600)

    def test_raise_range_context_uses_players_at_historical_action(self):
        from copy import deepcopy
        players = [dict(seat=i, stack=1000, committed=0, bet=0, folded=False, all_in=False) for i in range(8)]
        opening = {"seat": 3, "street": "preflop", "action": "raise", "amount": 60,
                   "stack_before": 1000, "is_all_in": False}
        before_folds = {"hand_id": 1, "hero": 0, "players": players, "big_blind": 20,
                        "history": [opening]}
        after_folds = deepcopy(before_folds)
        for seat in (4, 5, 6, 7, 1):
            after_folds["players"][seat]["folded"] = True
            after_folds["history"].append({"seat": seat, "street": "preflop", "action": "fold"})
        first, second = Agent(config={"adapt": False}), Agent(config={"adapt": False})
        first._update(before_folds)
        second._update(after_folds)
        self.assertEqual(first.opponents[3].width, second.opponents[3].width)

    def test_python_icm_matches_native_for_standard_chip_domain(self):
        if not native_available:
            self.skipTest("optional native backend unavailable")
        engine = NativeEvaluator()
        rng = random.Random(131117)
        for n in range(2, 9):
            for _ in range(6):
                stacks = [rng.randrange(0, 16001) for _ in range(n)]
                payouts = sorted((rng.random() for _ in range(n)), reverse=True)
                hero = rng.randrange(n)
                self.assertAlmostEqual(equity(stacks, payouts, hero), engine.icm_equity(stacks, payouts, hero), places=11)

    def test_training_corpus_first_1000_actions_legal_without_fallback(self):
        path = ROOT / "measurements" / "training_observations.jsonl.gz"
        if not path.exists():
            self.skipTest("optional local training corpus absent")
        agents = {}
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.assertLess(record["seed"], 900000, "audit uses training seeds only")
                key = record["seed"], record["table_id"]
                if key not in agents:
                    agents[key] = Agent(seed=record["seed"]*17)
                agent = agents[key]
                observation = record["observation"]
                validate(observation)
                decision = agent.act(observation)
                action = decision["action"]
                if action == "raise":
                    target = decision["amount"]
                    self.assertTrue(observation["can_raise"])
                    self.assertGreater(target, observation["current_bet"])
                    self.assertLessEqual(target, observation["max_raise_to"])
                    self.assertTrue(target >= observation["min_raise_to"] or target == observation["max_raise_to"])
                elif action == "check":
                    self.assertEqual(observation["to_call"], 0)
                elif action == "call":
                    self.assertGreater(observation["to_call"], 0)
                else:
                    self.assertEqual(action, "fold")
                    self.assertGreater(observation["to_call"], 0)
                self.assertEqual(agent.fallbacks, 0)
                count += 1
                if count == 1000:
                    break
        self.assertEqual(count, 1000)


if __name__ == "__main__":
    unittest.main()
