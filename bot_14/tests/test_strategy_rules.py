"""Behavioral and rule-boundary regressions for the actual local policy.

These fixtures use only the documented independent local schema. Synthetic
point distributions deliberately exercise ICM; they are not contest payouts.
"""

from __future__ import annotations

import copy
import random
import socket
import subprocess
import unittest
from unittest.mock import patch

from agent import Agent
from pokerbot.game import Table, icm_equities
from pokerbot.icm import call_threshold
from pokerbot.interface import card, legalize
from pokerbot.strategy import MODEL, NATIVE, TABLES


def observation(*, stacks=(500, 500), hero=0, board=(), hole=("As", "Kd"),
                invested=None, committed=None, folded=None, current_bet=0,
                hand_start=None, payouts=None, history=(), can_raise=True,
                small_blind_seat=0, big_blind_seat=1, button=0):
    n = len(stacks)
    invested = list(invested or [0] * n)
    committed = list(committed or invested)
    folded = list(folded or [False] * n)
    hand_start = list(hand_start or [stacks[i] + invested[i] for i in range(n)])
    result = {
        "hero": hero, "button": button,
        "small_blind_seat": small_blind_seat, "big_blind_seat": big_blind_seat,
        "hand_id": 8, "street": {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(board)],
        "hole_cards": tuple(card(c) if isinstance(c, str) else c for c in hole),
        "board": tuple(card(c) if isinstance(c, str) else c for c in board),
        "players": [dict(stack=stacks[i], invested=invested[i], committed=committed[i],
                         folded=folded[i], all_in=stacks[i] == 0,
                         starting_stack=hand_start[i]) for i in range(n)],
        "pot": sum(invested), "current_bet": current_bet,
        "to_call": max(0, current_bet - committed[hero]),
        "min_raise_to": current_bet + max(20, current_bet),
        "max_raise_to": stacks[hero] + committed[hero], "can_raise": can_raise,
        "big_blind": 20, "small_blind": 10,
        "history": copy.deepcopy(list(history)),
        "starting_stack": hand_start[hero], "hand_start_stacks": tuple(hand_start),
        "max_hands": 200,
    }
    if payouts is not None:
        result["payouts"] = tuple(payouts)
    return result


class StrategyBoundaryTests(unittest.TestCase):
    def test_short_all_in_big_blind_charges_only_matchable_call_cost(self):
        obs = observation(
            stacks=(490, 0), invested=(10, 15), committed=(10, 15), current_bet=20,
            hand_start=(500, 15), can_raise=False, hole=("7s", "2d"),
        )
        self.assertEqual(obs["to_call"], 10)  # Nominal full-BB bring-in.
        # Only five more chips can be matched. Equity .20 exceeds 5/30, but
        # would fail the incorrect unreturned-ten-chip threshold of 10/35.
        with patch.object(TABLES, "equity", return_value=0.20), \
                patch.object(TABLES, "nash_probability", side_effect=AssertionError("cannot raise into empty side pot")), \
                patch("pokerbot.strategy.call_threshold", wraps=call_threshold) as threshold:
            self.assertEqual(Agent(config={"nash": True}).decide(obs)["action"], "call")
            self.assertEqual(threshold.call_args.args[2], 5)

    def test_small_blind_facing_big_blind_shove_after_limp_avoids_wrong_nash_role(self):
        history = (
            dict(seat=0, street="preflop", action="post_sb", amount=10, paid=10, to=10),
            dict(seat=1, street="preflop", action="post_bb", amount=20, paid=20, to=20),
            dict(seat=0, street="preflop", action="call", amount=10, paid=10, to=20),
            dict(seat=1, street="preflop", action="raise", amount=200, paid=180, to=200),
        )
        obs = observation(
            stacks=(180, 0), invested=(20, 200), committed=(20, 200), current_bet=200,
            hand_start=(200, 200), can_raise=False, hole=("As", "Ad"), history=history,
        )
        with patch.object(TABLES, "nash_probability", return_value=0.0) as nash, \
                patch.object(TABLES, "equity", return_value=0.80):
            self.assertEqual(Agent(config={"nash": True}).decide(obs)["action"], "call")
            nash.assert_not_called()

    def test_optional_nash_uses_distinct_sb_and_bb_roles(self):
        blinds = [dict(seat=0, street="preflop", action="post_sb", amount=10, paid=10, to=10),
                  dict(seat=1, street="preflop", action="post_bb", amount=20, paid=20, to=20)]
        sb = observation(stacks=(190, 180), invested=(10, 20), committed=(10, 20),
                         current_bet=20, hand_start=(200, 200), history=blinds)
        with patch.object(TABLES, "nash_probability", return_value=1.) as policy:
            self.assertEqual(Agent(config={"nash": True}).decide(sb), {"action": "raise", "amount": 200})
            self.assertEqual(policy.call_args.args[-1], "sb")
        bb = observation(stacks=(0, 180), invested=(200, 20), committed=(200, 20),
                         current_bet=200, hand_start=(200, 200), hero=1, can_raise=False,
                         history=blinds + [dict(seat=0, street="preflop", action="raise", amount=200, paid=190, to=200)])
        with patch.object(TABLES, "nash_probability", return_value=1.) as policy:
            self.assertEqual(Agent(config={"nash": True}).decide(bb), {"action": "call"})
            self.assertEqual(policy.call_args.args[-1], "bb")

    def test_board_royal_tie_never_folds_to_multiway_all_in(self):
        obs = observation(
            stacks=(500, 0, 0), invested=(0, 1000, 1000), current_bet=1000,
            hole=("Kd", "3h"), board=("As", "Ks", "Qs", "Js", "Ts"),
            can_raise=False,
        )
        self.assertEqual(Agent().decide(obs)["action"], "call")

    def test_absolute_quads_never_folds_to_big_all_in(self):
        obs = observation(
            stacks=(200, 0, 0), invested=(0, 2000, 2000), current_bet=2000,
            hole=("Ac", "Ad"), board=("Ah", "As", "7d", "4c", "2h"),
            can_raise=False, payouts=(0.5, 0.3, 0.2),
        )
        self.assertEqual(Agent().decide(obs)["action"], "call")

    def test_board_ace_quads_and_dry_broadway_override_bad_equity_estimate(self):
        cases = (
            (("2d", "3s"), ("Ac", "Ad", "Ah", "As", "Kc")),
            (("Kh", "3s"), ("Ac", "Ad", "Ah", "As", "Qc")),
            (("2d", "3s"), ("Ac", "Kd", "Qh", "Js", "Tc")),
        )
        for hole, board in cases:
            with self.subTest(hole=hole, board=board):
                obs = observation(
                    stacks=(200, 0, 0), invested=(0, 2000, 2000), current_bet=2000,
                    hole=hole, board=board, can_raise=False, payouts=(0.5, 0.3, 0.2),
                )
                # The proof of an unbeatable hand takes priority over inference.
                with patch.object(MODEL, "predict", return_value=0.0):
                    self.assertEqual(Agent(config={"native": False}).decide(obs)["action"], "call")

    def test_no_private_opponent_fields_affect_decisions(self):
        obs = observation(stacks=(490, 480, 500), invested=(10, 20, 0), current_bet=20,
                          hero=2, small_blind_seat=0, big_blind_seat=1, button=2)
        one, two = copy.deepcopy(obs), copy.deepcopy(obs)
        one["players"][0]["hole_cards"] = [card("2c"), card("3d")]
        two["players"][0]["hole_cards"] = [card("Ac"), card("Ad")]
        one["private_deck"] = list(range(52))
        two["private_deck"] = list(reversed(range(52)))
        self.assertEqual(Agent(seed=917).decide(one), Agent(seed=917).decide(two))

    def test_postflop_private_fields_do_not_affect_native_or_portable_policy(self):
        obs = observation(
            stacks=(500, 300), invested=(100, 300), committed=(0, 200), current_bet=200,
            hole=("As", "Kh"), board=("Kc", "9d", "7h", "4s", "2c"), can_raise=False,
        )
        for native in (False, True):
            with self.subTest(native=native):
                one, two = copy.deepcopy(obs), copy.deepcopy(obs)
                one["players"][1]["hole_cards"] = (card("Ad"), card("Ah"))
                two["players"][1]["hole_cards"] = (card("3c"), card("5d"))
                one["private_deck"] = tuple(range(52))
                two["private_deck"] = tuple(reversed(range(52)))
                self.assertEqual(Agent(seed=917, config={"native": native}).decide(one),
                                 Agent(seed=917, config={"native": native}).decide(two))

    def test_postflop_decision_performs_no_python_network_process_or_file_calls(self):
        obs = observation(
            stacks=(500, 300), invested=(100, 300), committed=(0, 200), current_bet=200,
            hole=("As", "Kh"), board=("Kc", "9d", "7h", "4s", "2c"), can_raise=False,
        )
        for native in (False, True):
            agent = Agent(seed=918, config={"native": native})
            with self.subTest(native=native), \
                    patch.object(socket, "socket", side_effect=AssertionError("network during decision")), \
                    patch.object(socket, "create_connection", side_effect=AssertionError("network during decision")), \
                    patch.object(subprocess, "Popen", side_effect=AssertionError("process during decision")), \
                    patch("builtins.open", side_effect=AssertionError("file I/O during decision")):
                self.assertIn(agent.decide(obs)["action"], ("call", "fold"))

    def test_native_failure_uses_portable_model_and_caches_result(self):
        obs = observation(
            stacks=(500, 0), invested=(100, 300), committed=(0, 200), current_bet=200,
            hole=("As", "Kh"), board=("Kc", "9d", "7h"), can_raise=False,
        )
        with patch.object(NATIVE, "available", True), \
                patch.object(NATIVE, "estimate", return_value=None) as native_predictor, \
                patch.object(NATIVE, "predict_model", return_value=None) as native_forest, \
                patch.object(MODEL, "predict", return_value=0.75) as portable_predictor:
            agent = Agent()
            self.assertEqual(agent.decide(obs)["action"], "call")
            self.assertEqual(agent.decide(obs)["action"], "call")
            self.assertEqual(native_predictor.call_count, 1)
            self.assertEqual(native_forest.call_count, 1)
            self.assertEqual(portable_predictor.call_count, 1)
            self.assertEqual(agent.diagnostics["equity_native_calls"], 1)
            self.assertEqual(agent.diagnostics["equity_native_fallbacks"], 1)

    def test_native_equity_failure_can_use_native_forest_without_python_prediction(self):
        obs = observation(
            stacks=(500, 0), invested=(100, 300), committed=(0, 200), current_bet=200,
            hole=("As", "Kh"), board=("Kc", "9d", "7h"), can_raise=False,
        )
        with patch.object(NATIVE, "available", True), \
                patch.object(NATIVE, "estimate", return_value=None) as native_predictor, \
                patch.object(NATIVE, "predict_model", return_value=0.75) as native_forest, \
                patch.object(MODEL, "predict", side_effect=AssertionError("unnecessary Python fallback")):
            agent = Agent(config={"native_mix": 0.5})
            self.assertEqual(agent.decide(obs)["action"], "call")
            self.assertEqual(agent.decide(obs)["action"], "call")
            self.assertEqual(native_predictor.call_count, 1)
            self.assertEqual(native_forest.call_count, 1)
            self.assertEqual(agent.diagnostics["equity_model_native_calls"], 1)
            self.assertEqual(agent.diagnostics["equity_native_fallbacks"], 1)

    def test_reset_reproduces_initial_action_and_clears_public_history(self):
        obs = observation(stacks=(490, 480), invested=(10, 20), current_bet=20,
                          history=[dict(seat=1, action="raise", street="preflop", amount=20, paid=20)])
        agent = Agent(seed=916)
        first = agent.decide(obs)
        self.assertTrue(any(sum(stats) for stats in agent._stats))
        agent.reset()
        self.assertTrue(all(sum(stats) == 0 for stats in agent._stats))
        self.assertEqual(agent.decide(obs), first)
        self.assertEqual(agent.diagnostics["decisions"], 1)

    def test_eligible_pot_excludes_upper_side_pot(self):
        obs = observation(stacks=(100, 0, 500), invested=(100, 700, 500),
                          committed=(0, 400, 200), current_bet=400)
        # Hero's 100 call reaches 200 total; every contributor supplies at most
        # 200 to a pot hero can win. Current 1300-chip pot includes upper pots.
        self.assertEqual(Agent._eligible_pot(obs, 100), 500)

    def test_strong_heads_up_side_pot_is_not_charged_multiway_equity(self):
        obs = observation(
            stacks=(300, 0, 0), invested=(100, 50, 500), committed=(0, 0, 400),
            current_bet=400, hand_start=(400, 50, 500), can_raise=False,
            hole=("As", "Kh"), board=("Kc", "9d", "7h", "4s", "2c"),
        )
        def known_equity(features, q, m, opponents, **kwargs):
            return 0.70 if opponents == 1 else 0.20
        # Calling 300 contests a 150 main pot against two opponents and a 700
        # side pot against only the deep opponent. Even .20/.70 showdown shares
        # imply 520 expected returned chips, comfortably exceeding the 300 call.
        # Applying .20 to the whole 850 would incorrectly fold.
        with patch.object(MODEL, "predict", side_effect=known_equity) as predictor:
            self.assertEqual(Agent(config={"native": False}).decide(obs)["action"], "call")
            self.assertEqual(predictor.call_count, 2)

    def test_legalizer_returns_short_all_in_raise_to(self):
        obs = observation(stacks=(15, 300), invested=(30, 40), current_bet=40)
        obs["min_raise_to"] = 60
        self.assertEqual(legalize(obs, {"action": "raise", "amount": 1000}),
                         {"action": "raise", "amount": 45})
        obs["can_raise"] = False
        self.assertEqual(legalize(obs, {"action": "raise", "amount": 1000}), {"action": "fold"})

    def test_preflop_shove_range_uses_recorded_preflop_stack(self):
        # A 50BB open-shove stays 50BB after committed resets on a later street.
        event = dict(seat=1, action="raise", street="preflop", amount=1000,
                     paid=990, to=1000)
        pre = observation(stacks=(500, 0, 500), invested=(20, 1000, 20),
                          committed=(20, 1000, 20), current_bet=1000,
                          hand_start=(520, 1000, 520), history=[event])
        post = copy.deepcopy(pre)
        post.update(street="flop", board=(card("2c"), card("5d"), card("9h")),
                    current_bet=0, to_call=0)
        for player in post["players"]:
            player["committed"] = 0
        agent = Agent(config={"adaptation": False})
        q_pre = agent._ranges(pre, [0, 1, 2])[0][1]
        q_post = agent._ranges(post, [0, 1, 2])[0][1]
        self.assertAlmostEqual(q_pre, q_post)

    def test_postflop_all_in_does_not_reclassify_preflop_open_as_shove(self):
        event = dict(seat=1, action="raise", street="preflop", amount=50, paid=40, to=50)
        before = observation(stacks=(500, 950, 500), invested=(50, 50, 50),
                             committed=(50, 50, 50), current_bet=50,
                             hand_start=(550, 1000, 550), history=[event])
        after = copy.deepcopy(before)
        after.update(street="turn", board=tuple(card(c) for c in ("2c", "5d", "9h", "Jc")))
        after["players"][1].update(stack=0, committed=0, invested=1000, all_in=True)
        agent = Agent(config={"adaptation": False})
        self.assertAlmostEqual(agent._ranges(before, [0, 1, 2])[0][1],
                               agent._ranges(after, [0, 1, 2])[0][1])


class StrategyICMTests(unittest.TestCase):
    def test_previously_eliminated_seats_do_not_average_unrelated_bottom_prizes(self):
        obs = observation(
            stacks=(100, 0, 1, 0, 0, 0, 0, 0), invested=(0, 100, 0, 0, 0, 0, 0, 0),
            hand_start=(100, 100, 1, 0, 0, 0, 0, 0), current_bet=100,
            payouts=(8, 7, 6, 5, 4, 3, 2, 1), can_raise=False,
        )
        fold = icm_equities((100, 100, 1), (8, 7, 6))[0]
        lose = 6.0
        win = icm_equities((200, 1), (8, 7))[0]
        expected = max(0.5, (fold - lose) / (win - lose))
        self.assertAlmostEqual(call_threshold(obs, 1, 100, 100), expected)

    def test_two_remaining_players_have_linear_chip_threshold(self):
        obs = observation(
            stacks=(100, 0, 0, 0, 0, 0, 0, 0), invested=(0, 100, 0, 0, 0, 0, 0, 0),
            hand_start=(100, 100, 0, 0, 0, 0, 0, 0), current_bet=100,
            payouts=(8, 7, 6, 5, 4, 3, 2, 1), can_raise=False,
        )
        self.assertAlmostEqual(call_threshold(obs, 1, 100, 100), 0.5)

    def test_shove_icm_counts_future_caller_chips_once(self):
        obs = observation(
            stacks=(100, 80, 1), invested=(0, 20, 0), folded=(False, False, True),
            hand_start=(100, 100, 1), current_bet=20, payouts=(0.5, 0.3, 0.2),
        )
        # Caller will add 80. Conditional call outcomes are fold=(100,100,1),
        # win=(200,0,1), lose=(0,200,1); all contain the same 201 total chips.
        fold_value = icm_equities((100, 100, 1), (0.5, 0.3, 0.2))[0]
        win_value = icm_equities((200, 1), (0.5, 0.3))[0]
        lose_value = 0.2
        threshold = (fold_value - lose_value) / (win_value - lose_value)
        expected = -(threshold - 0.5) * 200
        with patch.object(TABLES, "mass", return_value=1.0), patch.object(TABLES, "equity", return_value=0.5):
            actual = Agent(config={"adaptation": False})._shove_ev(obs, [1], 0)
        self.assertAlmostEqual(actual, expected)


class ActualPolicyEngineTests(unittest.TestCase):
    def test_policy_legality_and_conservation_across_table_sizes(self):
        # Independent engine validates every actual output, including repeated
        # raises, short all-ins, dead small blinds and heads-up transitions.
        decisions = 0
        rng = random.Random(140081)
        for n in range(2, 9):
            for seed in range(10):
                stacks = [rng.randint(25, 600) for _ in range(n)]
                agents = [Agent(seed=1400000 + 10 * seed + seat) for seat in range(n)]
                table = Table(stacks, seed=9300 + seed, small_blind=5, big_blind=10,
                              payouts=tuple(float(n - place) for place in range(n)),
                              max_hands=80, strict=True)
                for hand in range(80):
                    if sum(stack > 0 for stack in table.stacks) <= 1:
                        break
                    table.set_blinds(5 * 2 ** (hand // 20), 10 * 2 ** (hand // 20))
                    result = table.play_hand(agents)
                    self.assertEqual(sum(result.ending_stacks), sum(stacks))
                    self.assertTrue(all(stack >= 0 for stack in result.ending_stacks))
                    self.assertEqual(result.errors, ())
                decisions += table.decision_count
                self.assertEqual(table.invalid_actions, 0)
                self.assertEqual(table.exceptions, 0)
        self.assertGreater(decisions, 5000)


if __name__ == "__main__":
    unittest.main()
