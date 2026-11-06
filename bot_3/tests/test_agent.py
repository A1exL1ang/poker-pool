import copy
import random
import unittest

from bot3.agent import Agent
from bot3.cards import card_id
from bot3.equity import EquityModel
from bot3.interface import validate_action, validate_observation
from tools.baselines import ScriptedOpponent
from tools.simulator import Table


def fixture(hole=(48, 49), board=(), n=8, stack=1980, call=20, pot=30, seat=2):
    ps = [{"seat": i, "stack": stack, "bet": 0, "committed": 0, "folded": False, "all_in": False} for i in range(n)]
    ps[0].update(bet=10, committed=10)
    ps[1].update(bet=call, committed=call)
    street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)]
    return {"table_id": "test", "hand_id": 1, "hero_seat": seat, "button": n - 1,
            "hole_cards": list(hole), "board": list(board), "street": street,
            "players": ps, "pot": pot, "big_blind": 20, "to_call": call,
            "min_raise_to": call + 20, "max_raise_to": stack, "legal_actions": ["fold", "call", "raise"] if call else ["check", "raise"], "history": []}


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EquityModel()

    def test_aa_open_and_junk_fold(self):
        a = Agent(model=self.model)
        o = fixture()
        self.assertEqual(a.act(o)["action"], "raise")
        o["hole_cards"] = [0, 21]  # 72o under the gun
        self.assertEqual(a.act(o)["action"], "fold")

    def test_short_stack_aa_commits(self):
        a = Agent(model=self.model)
        o = fixture(stack=180)
        self.assertEqual(a.act(o), {"action": "raise", "amount": 180})

    def test_free_check_never_folds(self):
        a = Agent(model=self.model)
        rng = random.Random(3)
        for _ in range(100):
            cs = rng.sample(range(52), 7)
            o = fixture(cs[:2], cs[2:], call=0)
            o["players"][0].update(bet=0, committed=0)
            self.assertIn(a.act(o)["action"], ("check", "raise"))

    def test_input_immutable(self):
        a = Agent(model=self.model)
        o = fixture()
        before = copy.deepcopy(o)
        a.act(o)
        self.assertEqual(o, before)

    def test_shared_board_royal_flush_call(self):
        a = Agent(model=self.model)
        board = [card_id(c) for c in ("As", "Ks", "Qs", "Js", "Ts")]
        o = fixture((0, 5), board, call=100, pot=300, stack=100)
        self.assertEqual(a.act(o)["action"], "call")

    def test_nut_royal_flush_never_folds(self):
        a = Agent(model=self.model)
        board = [card_id(c) for c in ("Qs", "Js", "Ts", "2d", "3c")]
        o = fixture((card_id("As"), card_id("Ks")), board, call=2000, pot=2030, stack=2000)
        self.assertNotEqual(a.act(o)["action"], "fold")

    def test_supplied_flat_payout_bubble_prevents_premium_reraise(self):
        a = Agent(model=self.model)
        o = fixture((48, 49), n=3, stack=1000, call=900, pot=900)
        o["players"][0].update(stack=1, bet=0, committed=0, folded=True)
        o["players"][1].update(stack=1100)
        o["payouts"] = [1, 1]
        o["history"] = [{"seat": 1, "street": "preflop", "action": "raise", "amount": 900}]
        action = a.act(o)
        self.assertGreater(a.last_info["icm_premium"], .1)
        self.assertNotEqual(action["action"], "raise")

    def test_private_royal_not_folded_due_to_binary_icm(self):
        a = Agent(model=self.model)
        board = [card_id(c) for c in ("Qs", "Js", "Ts", "2d", "3c")]
        o = fixture((card_id("As"), card_id("Ks")), board, n=3, stack=1000, call=1000, pot=1000)
        o["players"][0].update(stack=1, bet=0, committed=0, folded=True)
        o["players"][1].update(stack=0, all_in=True)
        o["payouts"] = [1, 1]
        o["legal_actions"] = ["fold", "call"]
        self.assertEqual(a.act(o)["action"], "call")

    def test_shared_board_ties_split_instead_of_compound_as_wins(self):
        from bot3.cards import features
        class TieModel:
            def postflop(self, hole, board):
                return (.5,) * 6, features(hole, board)
        a = Agent(model=TieModel())
        board = [card_id(c) for c in ("As", "Ad", "Ah", "Ks", "Kd")]
        o = fixture((0, 5), board, n=4, call=0)
        o["players"][0].update(bet=0, committed=0)
        a.act(o)
        self.assertAlmostEqual(a.last_info["equity"], .25)

    def test_state_resets_per_table_and_history_deduplicated(self):
        a = Agent(model=self.model)
        o = fixture()
        o["history"] = [{"seat": 4, "street": "preflop", "action": "raise", "amount": 60}]
        a.act(o)
        a.act(o)
        self.assertEqual(a.stats[4][0], 1)
        o["table_id"] = "new"
        o["history"] = []
        a.act(o)
        self.assertEqual(a.stats, {})

    def test_randomized_legal_decisions(self):
        rng = random.Random(120)
        a = Agent(model=self.model)
        for k in range(3000):
            nb = rng.choice((0, 3, 4, 5))
            cs = rng.sample(range(52), nb + 2)
            stack = rng.randrange(1, 20001)
            call = rng.randrange(0, 15000)
            o = fixture(cs[:2], cs[2:], stack=stack, call=call, pot=call * 2 + rng.randrange(1, 2000))
            o["players"][0].update(bet=0, committed=0)
            o["to_call"] = min(stack, call)
            if call == 0:
                o["legal_actions"] = ["check", "raise"]
            elif stack <= call:
                o["legal_actions"] = ["fold", "call"]
            validate_observation(o)
            validate_action(o, a.act(o))

    def test_tournament_conservation_and_legality(self):
        for seed in range(12):
            agents = [Agent(seed=seed + 3, model=self.model)] + [ScriptedOpponent(self.model, style=s, seed=seed * 8 + i) for i, s in enumerate(("balanced", "pressure", "tight", "passive", "random", "balanced", "pressure"))]
            table = Table(agents, seed=seed)
            result = table.run()
            self.assertEqual(sum(result["stacks"]), 16000)
            self.assertLessEqual(result["hands"], 200)


if __name__ == "__main__":
    unittest.main()
