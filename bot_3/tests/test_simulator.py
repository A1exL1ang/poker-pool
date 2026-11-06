import unittest
from unittest.mock import patch
from bot3.cards import card_id
from tools.simulator import award_pots, Table


def cs(s):
    return [card_id(x) for x in s.split()]


class Script:
    def __init__(self, moves):
        self.moves = iter(moves)
        self.observations = []

    def act(self, o):
        self.observations.append(o)
        try:
            return next(self.moves)
        except StopIteration:
            return {"action": "call", "amount": o["to_call"]} if o["to_call"] else {"action": "check", "amount": 0}


class SimulatorTests(unittest.TestCase):
    def test_side_pots_and_uncalled_return(self):
        ps = [{"committed": n, "folded": False, "stack": 0} for n in (50, 100, 200)]
        board = cs("2c 3d 7h 9s Jc")
        holes = [cs("As Ad"), cs("Ks Kd"), cs("Qs Qd")]
        self.assertEqual(award_pots(ps, board, 0, holes), [150, 100, 100])
        self.assertEqual(sum(p["stack"] for p in ps), 350)

    def test_folded_chips_and_odd_chip(self):
        ps = [{"committed": 5, "folded": i == 2, "stack": 0} for i in range(3)]
        board = cs("As Ks Qs Js Ts")
        holes = [cs("2c 3c"), cs("4c 5c"), cs("6c 7c")]
        self.assertEqual(award_pots(ps, board, 0, holes), [7, 8, 0])

    def setup_round(self, stacks, bets, scripts, current=100):
        agents = [Script(moves) for moves in scripts]
        t = Table(agents, starting_stack=1000)
        t.holes = [[i * 2, i * 2 + 1] for i in range(len(stacks))]
        t.history = []
        for i, p in enumerate(t.players):
            p.update(stack=stacks[i], bet=bets[i], committed=bets[i], folded=False, all_in=stacks[i] == 0)
        return t, agents

    def test_short_raise_does_not_reopen(self):
        # A checks at 100; B raises all-in to 150 (less than full 100 raise).
        t, agents = self.setup_round([900, 50, 900], [100, 100, 100], [[], [{"action": "raise", "amount": 150}], []])
        t.betting(0, [], "preflop", 100, 100)
        self.assertEqual(len(agents[0].observations), 2)
        self.assertNotIn("raise", agents[0].observations[-1]["legal_actions"])
        self.assertEqual([p["bet"] for p in t.players], [150, 150, 150])

    def test_full_raise_reopens(self):
        t, agents = self.setup_round([900, 900, 900], [100] * 3, [[], [{"action": "raise", "amount": 200}], []])
        t.betting(0, [], "preflop", 100, 100)
        self.assertIn("raise", agents[0].observations[-1]["legal_actions"])

    def test_cumulative_short_raises_reopen(self):
        t, agents = self.setup_round([900, 40, 100, 900], [100] * 4,
                                   [[], [{"action": "raise", "amount": 140}], [{"action": "raise", "amount": 200}], []])
        t.betting(0, [], "preflop", 100, 100)
        self.assertIn("raise", agents[0].observations[-1]["legal_actions"])

    def test_heads_up_button_first_preflop_last_postflop(self):
        agents = [Script([]), Script([])]
        t = Table(agents, starting_stack=2000, seed=10)
        t.hand()
        self.assertEqual(t.history[0]["seat"], t.button)
        flop = [e for e in t.history if e["street"] == "flop"]
        self.assertEqual(flop[0]["seat"], 1 - t.button)

    def test_short_big_blind_bring_in(self):
        t, agents = self.setup_round([100, 100, 0], [0, 10, 7], [[], [], []])
        t.betting(0, [], "preflop", 20, 20)
        first = agents[0].observations[0]
        self.assertEqual(first["current_bet"], 20)
        self.assertEqual(first["to_call"], 20)
        self.assertEqual(first["min_raise_to"], 40)

    def test_heads_up_short_blind_no_unmatched_call(self):
        t, agents = self.setup_round([100, 0], [10, 15], [[], []])
        t.betting(0, [], "preflop", 20, 20)
        self.assertEqual(agents[0].observations[0]["to_call"], 5)

    def test_timeout_falls_back_to_check_or_fold(self):
        t, _ = self.setup_round([900, 900], [100, 100], [[{"action": "raise", "amount": 200}], []])
        t.decision_budget_ns = 1000000
        with patch("tools.simulator.time.perf_counter_ns", side_effect=range(0, 20000000, 2000000)):
            t.betting(0, [], "preflop", 100, 100)
        self.assertEqual(t.timeouts, [1, 1])
        self.assertEqual([e["action"] for e in t.history], ["check", "check"])
        t, _ = self.setup_round([900, 900], [0, 100], [[], []])
        t.decision_budget_ns = 1000000
        with patch("tools.simulator.time.perf_counter_ns", side_effect=range(0, 20000000, 2000000)):
            t.betting(0, [], "preflop", 100, 100)
        self.assertEqual(t.history[0]["action"], "fold")
