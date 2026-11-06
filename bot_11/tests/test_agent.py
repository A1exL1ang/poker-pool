"""Policy contracts and strategic edge cases, independent of trained assets.

Small controlled equity predictions isolate mechanics from model quality.
Real-asset end-to-end/timing validation lives in tools.latency and benchmark.
"""
from collections import defaultdict
from copy import deepcopy
import random
import unittest
from unittest.mock import patch

import pokerbot.agent as policy
from pokerbot.evaluator import parse_cards
from pokerbot.engine import Hand


class ConstantModel:
    def __init__(self, predictions=None):
        self.predictions = predictions or [.9, .82, .75, .87, .84, .81, .83, .77, .80, .73]
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        return self.predictions[:]


def row():
    return {"percentile": .50, "equity": [.55, .49, .45, .41, .37, .33, .29],
            "uniform2": .36, "uniform3": .27}


def observation(hole="Ac Ad", board="", stacks=None, commitments=None,
                street_bets=None, call=0, can_raise=True, folded=None,
                payouts=None, history=None, hand_number=0):
    stacks = list(stacks or [1000, 1000])
    n = len(stacks)
    commitments = list(commitments or [20] * n)
    street_bets = list(street_bets if street_bets is not None else commitments)
    folded = list(folded or [False] * n)
    cards = list(parse_cards(board)) if board else []
    street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(cards)]
    current = max(street_bets)
    return {"hole_cards": list(parse_cards(hole)), "board": cards, "seat": 0,
            "button": 0, "small_blind": 10, "big_blind": 20,
            "small_blind_seat": 0 if n == 2 else 1,
            "big_blind_seat": 1 if n == 2 else 2,
            "hand_number": hand_number, "street": street,
            "players": [{"seat": i, "stack": stacks[i], "committed": commitments[i],
                         "street_bet": street_bets[i], "folded": folded[i],
                         "all_in": stacks[i] == 0 and not folded[i]} for i in range(n)],
            "pot": sum(commitments), "to_call": call, "current_bet": current,
            "min_raise_to": current + 20, "max_raise_to": street_bets[0] + stacks[0],
            "can_raise": can_raise,
            "legal_actions": ["fold", "call" if call else "check"] + (["raise"] if can_raise else []),
            "payouts": payouts, "history": history or []}


def passive(obs):
    return {"action": "call" if obs["to_call"] else "check"}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.model = ConstantModel()
        self.pf = defaultdict(row)
        self.patches = [patch.object(policy, "load_assets"),
                        patch.object(policy, "_MODEL", self.model),
                        patch.object(policy, "_PREFLOP", self.pf)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.agent = policy.Agent(seed=111)

    def test_forced_board_chops_call_even_at_break_even_price(self):
        self.model.predictions = [.001] * 10
        boards = ("Ts Js Qs Ks As", "Ac Ad Ah As Kc", "Kc Kd Kh Ks Ac", "As Kd Qh Jc Ts")
        for board in boards:
            for payouts in (None, [.7, .3]):
                with self.subTest(board=board, payouts=payouts):
                    o = observation(hole="2c 3d", board=board, stacks=[1000, 0],
                                    commitments=[0, 1000], street_bets=[0, 1000],
                                    call=1000, can_raise=False, payouts=payouts)
                    self.assertEqual(self.agent.act(o), {"action": "call"})

    def test_forced_chop_side_pot_layers(self):
        self.model.predictions = [.001] * 10
        o = observation(hole="2c 3d", board="Ts Js Qs Ks As", stacks=[100, 0, 0],
                        commitments=[0, 100, 50], street_bets=[0, 100, 50],
                        call=100, can_raise=False)
        self.assertEqual(self.agent.act(o), {"action": "call"})

    def test_owned_royal_flush_calls_despite_bad_model_prediction(self):
        self.model.predictions = [.001] * 10
        for hole, board in (("As Ks", "Qs Js Ts"),
                            ("As Ks", "Qs Js Ts 2c"),
                            ("As 2d", "Ks Qs Js Ts 3c")):
            with self.subTest(hole=hole, board=board):
                o = observation(hole=hole, board=board, stacks=[1000, 0],
                                commitments=[0, 1000], street_bets=[0, 1000],
                                call=1000, can_raise=False, payouts=[.7, .3])
                self.assertEqual(self.agent.act(o), {"action": "call"})

    def test_inaccessible_overbet_does_not_inflate_call_pot_odds(self):
        self.model.predictions = [.20] * 10
        o = observation(board="2c 3d 7h 9s Jc", stacks=[100, 0],
                        commitments=[50, 1000], street_bets=[0, 950],
                        call=100, can_raise=False)
        # Only150 of villain's1000 is contestable by hero after this call.
        # Using the raw1050-chip pot would make an unprofitable call look cheap.
        self.assertEqual(self.agent.act(o), {"action": "fold"})

    def test_allin_blind_call_uses_pot_odds_not_opening_range(self):
        self.pf["72o"] = {"percentile": .999, "equity": [.30] * 7,
                           "uniform2": .25, "uniform3": .17}
        o = observation(hole="7c 2d", stacks=[100, 0], commitments=[10, 20],
                        street_bets=[10, 20], call=10, can_raise=False)
        self.assertEqual(self.agent.act(o), {"action": "call"})
        self.pf["72o"]["equity"] = [.20] * 7
        self.assertEqual(self.agent.act(o), {"action": "fold"})

    def test_allin_blind_multiway_uses_multiway_equity(self):
        self.pf["72o"] = {"percentile": .999, "equity": [.30] * 7,
                           "uniform2": .25, "uniform3": .17}
        o = observation(hole="7c 2d", stacks=[100, 0, 0], commitments=[10, 20, 20],
                        street_bets=[10, 20, 20], call=10, can_raise=False)
        self.assertEqual(self.agent.act(o), {"action": "call"})
        self.pf["72o"]["uniform2"] = .10
        self.assertEqual(self.agent.act(o), {"action": "fold"})

    def test_raise_to_semantics_and_minimum_clamping(self):
        o = observation(stacks=[1000, 1000], commitments=[50, 100], street_bets=[50, 100], call=50)
        o["min_raise_to"] = 200
        action = self.agent._legalize(o, o["players"][0], {"action": "raise", "amount": 150})
        self.assertEqual(action, {"action": "raise", "amount": 200})
        action = self.agent._legalize(o, o["players"][0], {"action": "raise", "amount": 99999})
        self.assertEqual(action, {"action": "raise", "amount": 1050})

    def test_explicit_icm_risk_also_restricts_premium_reraise(self):
        self.pf['QQ'] = {'percentile':.01, 'equity':[.55]*7, 'uniform2':.3, 'uniform3':.2}
        o = observation(hole='Qc Qd', stacks=[1000,1000]+[1]*6,
                        commitments=[100,700]+[0]*6, street_bets=[100,700]+[0]*6,
                        call=600, folded=[False,False]+[True]*6,
                        payouts=[.5,.3,.2,0,0,0,0,0], history=[
                            {'seat':0,'street':'preflop','action':'raise','amount':100},
                            {'seat':1,'street':'preflop','action':'raise','amount':700}])
        with patch.object(policy,'call_threshold',return_value=.90):
            self.assertEqual(self.agent.act(o),{'action':'fold'})
        o['payouts'] = None
        self.assertEqual(self.agent.act(o)['action'],'raise')

    def test_short_allin_raise_and_closed_raising(self):
        o = observation(stacks=[50, 1000], commitments=[100, 100], street_bets=[100, 100])
        o["min_raise_to"] = 200
        self.assertEqual(self.agent._legalize(o, o["players"][0], {"action": "raise", "amount": 200}),
                         {"action": "raise", "amount": 150})
        o["can_raise"] = False
        o["legal_actions"] = ["fold", "call"]
        o["to_call"] = 20
        self.assertEqual(self.agent._legalize(o, o["players"][0], {"action": "raise", "amount": 150}),
                         {"action": "call"})

    def test_check_instead_of_folding_free_action(self):
        o = observation()
        self.assertEqual(self.agent._legalize(o, o["players"][0], {"action": "fold"}), {"action": "check"})

    def test_malformed_input_fallbacks(self):
        inputs = [None, {}, {"legal_actions": ["check"]}]
        duplicate = observation()
        duplicate["hole_cards"] = [0, 0]
        inputs.append(duplicate)
        invalid = observation()
        invalid["hole_cards"] = [-1, 52]
        inputs.append(invalid)
        street = observation()
        street["street"] = "river"
        inputs.append(street)
        river_with_flop = observation(board="2c 3d 7h")
        river_with_flop["street"] = "river"
        inputs.append(river_with_flop)
        missing_hero = observation()
        missing_hero["seat"] = 999
        inputs.append(missing_hero)
        for o in inputs:
            with self.subTest(input=o):
                legal = o.get("legal_actions", []) if isinstance(o, dict) else []
                expected = {"action": "check" if "check" in legal else "fold"}
                self.assertEqual(self.agent.act(o), expected)

    def test_reset_removes_table_opponent_and_cache_state(self):
        o = observation(board="2c 3d 7h", history=[{"seat": 1, "street": "preflop", "action": "raise", "amount": 60}])
        self.agent.act(o)
        self.assertTrue(self.agent.stats)
        self.assertTrue(self.agent._cache)
        self.agent.reset()
        self.assertEqual(self.agent.stats, {})
        self.assertEqual(self.agent._cache, {})
        self.assertIsNone(self.agent._hand)
        self.assertEqual(self.agent._seen, 0)
        self.assertEqual(self.agent._vpip_seen, set())
        self.assertEqual(self.agent._pfr_seen, set())

    def test_reset_restores_seeded_random_state(self):
        initial = policy.Agent(seed=111)._random()
        self.agent._random()
        self.agent._random()
        self.agent.reset()
        self.assertEqual(self.agent._random(), initial)

    def test_history_is_processed_once_and_new_hand_clears_cache(self):
        o = observation(board="2c 3d 7h", history=[{"seat": 1, "street": "preflop", "action": "raise", "amount": 60}])
        self.agent.act(o)
        before = deepcopy(self.agent.stats)
        self.agent.act(o)
        self.assertEqual(self.agent.stats, before)
        self.assertEqual(self.model.calls, 1)
        o["hand_number"] = 1
        self.agent.act(o)
        self.assertEqual(self.model.calls, 2)

    def test_decision_has_no_file_or_network_access_after_setup(self):
        o = observation(board="2c 3d 7h")
        with patch("builtins.open", side_effect=AssertionError("gameplay file access")), \
             patch("socket.socket", side_effect=AssertionError("gameplay network access")):
            action = self.agent.act(o)
        self.assertIn(action["action"], o["legal_actions"])

    def test_string_cards_and_integer_cards_have_identical_decisions(self):
        o = observation(hole="Ac Ad", board="2c 3d 7h")
        text = deepcopy(o)
        text["hole_cards"] = ["Ac", "Ad"]
        text["board"] = ["2c", "3d", "7h"]
        a, b = policy.Agent(seed=717), policy.Agent(seed=717)
        self.assertEqual(a.act(o), b.act(text))

    def test_local_engine_private_observations_and_legal_actions(self):
        seen = []
        def candidate(o):
            seen.append(o)
            return self.agent.act(o)
        for seed in range(30):
            self.agent.reset()
            h = Hand([100 + 17 * seed] * 8, [candidate] + [passive] * 7,
                     rng=random.Random(seed), button=seed % 8)
            result = h.play()
            self.assertEqual(result["errors"], 0)
            self.assertEqual(result["invalid_actions"], 0)
        self.assertTrue(seen)
        for o in seen:
            self.assertTrue(all("hole_cards" not in p and "cards" not in p for p in o["players"]))
        self.assertEqual(self.agent.diagnostics["errors"], 0)


if __name__ == "__main__":
    unittest.main()
