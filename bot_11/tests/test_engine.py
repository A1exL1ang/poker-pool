"""Mechanics tests for the independent local simulator (not an official kit)."""
import unittest

from pokerbot.engine import Hand, Tournament, icm_values
from pokerbot.baselines import CallingStation, FoldAgent, make_baseline


def card(text):
    return "23456789TJQKA".index(text[0]) * 4 + "cdhs".index(text[1])


def fixed_deck(hands, board, button=0):
    n = len(hands)
    order = [(button + i) % n for i in range(1, n + 1)]
    holes = [[card(c) for c in hand.split()] for hand in hands]
    board = [card(c) for c in board.split()]
    used = {c for hand in holes for c in hand} | set(board)
    spare = iter(c for c in range(52) if c not in used)
    prefix = [holes[i][r] for r in range(2) for i in order]
    prefix += [next(spare)] + board[:3] + [next(spare), board[3], next(spare), board[4]]
    return prefix + list(spare)


def call(obs):
    return {"action": "call" if obs["to_call"] else "check"}


def jam(obs):
    return {"action": "raise", "amount": obs["max_raise_to"]} if obs["can_raise"] else call(obs)


class EngineTests(unittest.TestCase):
    def test_heads_up_order_and_observation_privacy(self):
        seen = []
        def observe(obs):
            seen.append(obs)
            return call(obs)
        h = Hand([100, 100], [observe, observe], button=0,
                 deck=fixed_deck(["Ac Ad", "Kc Kd"], "2c 3d 7h 9s Jc"))
        result = h.play()
        actions = [a for a in result["history"] if a["action"] not in ("blind", "return")]
        self.assertEqual([a["seat"] for a in actions[:2]], [0, 1])
        self.assertEqual([a["seat"] for a in actions[2:4]], [1, 0])
        self.assertEqual(seen[0]["hole_cards"], [card("Ac"), card("Ad")])
        self.assertTrue(all("cards" not in p and "hole_cards" not in p for p in seen[0]["players"]))
        self.assertEqual(result["stacks"], [120, 80])

    def test_multiway_side_pots_and_return(self):
        h = Hand([100, 200, 300], [jam] * 3, button=0,
                 deck=fixed_deck(["Ac Ad", "Kc Kd", "Qc Qd"], "2c 3d 7h 9s Jc"))
        result = h.play()
        self.assertEqual(result["stacks"], [300, 200, 100])
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["invalid_actions"], 0)

    def test_uncalled_raise_and_big_blind_return(self):
        h = Hand([100, 100], [FoldAgent(), CallingStation()], button=0)
        result = h.play()
        self.assertEqual(result["stacks"], [90, 110])
        returns = [a["amount"] for a in result["history"] if a["action"] == "return"]
        self.assertEqual(returns, [10])

    def test_odd_chip_left_of_button_with_folded_contribution(self):
        h = Hand([100] * 3, [call, FoldAgent(), call], button=0, small_blind=1, big_blind=3,
                 deck=fixed_deck(["2c 3d", "4c 5d", "6c 7d"], "Ts Js Qs Ks As"))
        result = h.play()
        self.assertEqual(result["stacks"], [100, 99, 101])

    def test_short_raise_does_not_reopen(self):
        seen = []
        def opener(obs):
            if obs["street"] == "preflop":
                seen.append(obs)
                if len(seen) == 1:
                    return {"action": "raise", "amount": 100}
            return call(obs)
        def short(obs):
            return jam(obs) if obs["street"] == "preflop" else call(obs)
        result = Hand([1000, 150, 1000, 1000], [opener, short, call, call], button=1).play()
        self.assertEqual(seen[1]["current_bet"], 150)
        self.assertFalse(seen[1]["can_raise"])
        self.assertEqual(seen[1]["min_raise_to"], 230)
        self.assertEqual(result["invalid_actions"], 0)

    def test_cumulative_short_raises_reopen(self):
        seen = []
        def opener(obs):
            if obs["street"] == "preflop":
                seen.append(obs)
                if len(seen) == 1:
                    return {"action": "raise", "amount": 100}
                if len(seen) == 2:
                    return {"action": "raise", "amount": 260}
            return call(obs)
        result = Hand([1000, 140, 180, 1000], [opener, jam, jam, call], button=1).play()
        self.assertEqual(seen[1]["current_bet"], 180)
        self.assertTrue(seen[1]["can_raise"])
        self.assertEqual(seen[1]["min_raise_to"], 260)
        self.assertEqual(result["invalid_actions"], 0)

    def test_check_then_short_opening_bet_does_not_reopen(self):
        seen = []
        def player_one(obs):
            if obs["street"] == "flop":
                seen.append(obs)
            return call(obs)
        def player_two(obs):
            return jam(obs) if obs["street"] == "flop" else call(obs)
        result = Hand([1000, 1000, 25], [call, player_one, player_two], button=0).play()
        self.assertEqual(seen[1]["current_bet"], 5)
        self.assertFalse(seen[1]["can_raise"])
        self.assertEqual(seen[1]["min_raise_to"], 25)
        self.assertEqual(result["invalid_actions"], 0)

    def test_short_opening_bet_raises_by_bb_not_completion(self):
        seen = []
        def unacted(obs):
            if obs["street"] == "flop":
                seen.append(obs)
                if len(seen) == 1:
                    return {"action": "raise", "amount": 25}
            return call(obs)
        def short(obs):
            return jam(obs) if obs["street"] == "flop" else call(obs)
        result = Hand([1000, 25, 1000], [call, short, unacted], button=0).play()
        self.assertTrue(seen[0]["can_raise"])
        self.assertEqual(seen[0]["min_raise_to"], 25)
        self.assertEqual(result["invalid_actions"], 0)

    def test_short_big_blind_heads_up_dry_sidepot(self):
        seen = []
        def observe(obs):
            seen.append(obs)
            return call(obs)
        result = Hand([100, 5], [observe, observe], button=0,
                      deck=fixed_deck(["Ac Ad", "Kc Kd"], "2c 3d 7h 9s Jc")).play()
        self.assertEqual(seen, [])
        self.assertEqual(result["stacks"], [105, 0])

    def test_allin_runs_out_board_without_extra_decisions(self):
        result = Hand([20] * 8, [jam] * 8, button=0).play()
        self.assertEqual(len(result["board"]), 5)
        self.assertEqual(sum(result["stacks"]), 160)
        self.assertEqual(result["decisions"], 7)  # BB is already all-in; SB still calls.

    def test_error_timeout_and_illegal_action_fallback(self):
        def crash(obs):
            raise RuntimeError("expected test crash")
        result = Hand([100, 100], [crash, call], button=0).play()
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["stacks"], [90, 110])
        result = Hand([100, 100], [call, call], button=0,
                      enforce_time_limit=True, decision_limit_ns=0).play()
        self.assertEqual(result["timeouts"], 1)
        result = Hand([100, 100], [lambda obs: {"action": "raise", "amount": 21}, call], button=0).play()
        self.assertEqual(result["invalid_actions"], 1)
        self.assertEqual(result["stacks"], [90, 110])

    def test_icm_symmetry_two_player_and_conservation(self):
        self.assertEqual(icm_values([100, 300], [100, 0]), [25, 75])
        self.assertEqual(icm_values([100, 300], [100, 50]), [62.5, 87.5])
        self.assertEqual(icm_values([100, 100, 100], [90, 30, 0]), [40, 40, 40])
        values = icm_values([1, 3, 7, 9, 15, 23, 31, 44], [100, 60, 30, 10, 0, 0, 0, 0])
        self.assertAlmostEqual(sum(values), 200)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        self.assertEqual(icm_values([0, 100], [100]), [0, 100])

    def test_tournament_cap_icm_and_default_no_payouts(self):
        result = Tournament([call] * 8, seed=13, max_hands=1,
                            payouts=[100, 60, 30, 10, 0, 0, 0, 0]).run()
        self.assertEqual(result["hands"], 1)
        self.assertAlmostEqual(sum(result["points"]), 200)
        self.assertAlmostEqual(sum(result["expected_ranks"]), 36)
        self.assertEqual(sum(result["stacks"]), 16000)
        self.assertIsNone(Tournament([call] * 2, max_hands=1).run()["points"])

    def test_simultaneous_elimination_tie_places(self):
        # Deterministic equal stacks all-in on first hand. Winner(s) may tie,
        # but all busted equal starting stacks must share their finish rank.
        result = Tournament([jam] * 8, seed=823, stacks=[20] * 8,
                            max_hands=1, payouts=list(range(8, 0, -1))).run()
        eliminated = [i for i, s in enumerate(result["stacks"]) if not s]
        self.assertTrue(eliminated)
        self.assertEqual(len({result["ranks"][i] for i in eliminated}), 1)
        self.assertAlmostEqual(sum(result["points"]), 36)

    def test_random_style_tournaments_conserve_and_actions_legal(self):
        names = ["fold", "station", "random", "tight_passive", "tag", "lag", "pushfold", "tag"]
        for seed in range(12):
            with self.subTest(seed=seed):
                agents = [make_baseline(name, seed * 11 + i) for i, name in enumerate(names)]
                result = Tournament(agents, seed=seed, max_hands=80).run()
                self.assertEqual(sum(result["stacks"]), 16000)
                self.assertEqual(result["errors"], 0)
                self.assertEqual(result["invalid_actions"], 0)
                self.assertTrue(all(s >= 0 for s in result["stacks"]))

    def test_big_blind_advances_across_eliminations(self):
        for seed in range(6):
            result = Tournament([jam] * 8, seed=seed, initial_stack=100,
                                retain_history=True, max_hands=40).run()
            for previous, following in zip(result["history"], result["history"][1:]):
                old_bb = previous["big_blind_seat"]
                new_bb = following["big_blind_seat"]
                live = previous["stacks"]
                expected = next((old_bb + offset) % 8 for offset in range(1, 9) if live[(old_bb + offset) % 8])
                self.assertEqual(new_bb, expected)
                if sum(s > 0 for s in live) > 2:
                    self.assertEqual(following["small_blind_position"], old_bb)
                    self.assertEqual(following["small_blind_seat"], old_bb if live[old_bb] else None)

    def test_dead_small_blind_posts_only_big_blind(self):
        seen = []
        def observe(obs):
            seen.append(obs)
            return call(obs)
        result = Hand([100, 100, 0, 100], [observe] * 4, button=1,
                      blind_seats=(2, 3)).play()
        self.assertEqual(seen[0]["pot"], 20)
        self.assertEqual(seen[0]["seat"], 0)
        self.assertEqual(result["small_blind_seat"], None)
        self.assertEqual(result["big_blind_seat"], 3)
        self.assertEqual(sum(result["stacks"]), 300)


if __name__ == "__main__":
    unittest.main()
