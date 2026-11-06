"""Rule-critical mechanics tests with independently specified chip outcomes."""
import math
import random
import unittest

from simulator import HoldemHand, InvalidAction, icm_values, run_tournament
from baselines import AlwaysFold, CallingStation, RandomLegal, ShoveBot


def cards(text):
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return [4 * ranks.index(c[0]) + suits.index(c[1]) for c in text.split()]


def fixed_deck(holes, board, button=0):
    """Arrange a physical round-robin deal, with independent arbitrary burns."""
    used = set(c for h in holes for c in h) | set(board)
    assert sum(map(len, holes)) + len(board) == len(used)
    remaining = [c for c in range(52) if c not in used]
    order = [(button + k) % len(holes) for k in range(1, len(holes) + 1)]
    deck = [holes[i][r] for r in range(2) for i in order if holes[i]]
    for street in (board[:3], board[3:4], board[4:5]):
        if street:
            deck.append(remaining.pop(0))
            deck.extend(street)
    return deck + remaining


def call_check(hand):
    while not hand.done:
        hand.step({"action": "call" if hand.observation()["to_call"] else "check"})


class MechanicsTests(unittest.TestCase):
    def test_heads_up_blinds_and_order(self):
        h = HoldemHand([2000, 2000], 0)
        self.assertEqual((h.sb_seat, h.bb_seat, h.to_act), (0, 1, 0))
        h.step({"action": "call"})
        self.assertEqual(h.to_act, 1)
        self.assertEqual(h.legal_actions(), ["check", "raise"])
        h.step({"action": "check"})
        self.assertEqual((h.street, h.to_act), ("flop", 1))

    def test_eight_handed_order_and_bb_option(self):
        h = HoldemHand([2000] * 8, 0)
        self.assertEqual((h.sb_seat, h.bb_seat, h.to_act), (1, 2, 3))
        for expected in (3, 4, 5, 6, 7, 0, 1):
            self.assertEqual(h.to_act, expected)
            h.step({"action": "call"})
        self.assertEqual(h.to_act, 2)
        h.step({"action": "check"})
        self.assertEqual((h.street, h.to_act, h.pot), ("flop", 1, 160))

    def test_raise_is_street_total_not_increment(self):
        h = HoldemHand([2000] * 3, 0)
        h.step({"action": "raise", "amount": 60})
        self.assertEqual((h.stacks[0], h.contributions[0], h.current_bet, h.last_full_raise), (1940, 60, 60, 40))
        self.assertEqual(h.observation()["min_raise_to"], 100)
        self.assertEqual(h.observation()["to_call"], 50)
        self.assertEqual(h.observation()["pot"], 90)

    def test_invalid_raises_do_not_mutate(self):
        h = HoldemHand([2000] * 3, 0)
        original = (h.stacks[:], h.contributions[:], h.history[:])
        for target in (21, 39, 2001, 40.0, True, None):
            with self.assertRaises(InvalidAction):
                h.step({"action": "raise", "amount": target})
            self.assertEqual((h.stacks, h.contributions, h.history), original)
        with self.assertRaises(InvalidAction):
            h.step({"action": "check"})
        with self.assertRaises(InvalidAction):
            h.step({"action": "call"}, seat=1)

    def test_single_short_allin_does_not_reopen(self):
        # Button 3, blinds 0/1, opener 2. Raise 20->100 is a full +80.
        h = HoldemHand([1000, 1000, 1000, 140], 3)
        h.step({"action": "raise", "amount": 100})
        h.step({"action": "raise", "amount": 140})
        h.step({"action": "call"})
        h.step({"action": "call"})
        self.assertEqual(h.to_act, 2)
        self.assertEqual(h.observation()["to_call"], 40)
        self.assertNotIn("raise", h.legal_actions())
        self.assertEqual(h.observation()["min_raise_to"], 220)

    def test_cumulative_short_allins_reopen(self):
        h = HoldemHand([1000, 1000, 1000, 140, 180], 4)
        h.step({"action": "raise", "amount": 100})
        h.step({"action": "raise", "amount": 140})
        h.step({"action": "raise", "amount": 180})
        h.step({"action": "call"})
        h.step({"action": "call"})
        self.assertEqual(h.to_act, 2)
        self.assertIn("raise", h.legal_actions())
        self.assertEqual(h.observation()["min_raise_to"], 260)
        h.step({"action": "raise", "amount": 260})
        self.assertEqual(h.last_full_raise, 80)

    def test_new_full_raise_updates_minimum(self):
        h = HoldemHand([1000] * 4, 3)
        h.step({"action": "raise", "amount": 100})
        h.step({"action": "raise", "amount": 300})
        self.assertEqual(h.observation()["min_raise_to"], 500)
        h.step({"action": "call"})
        h.step({"action": "call"})
        self.assertIn("raise", h.legal_actions())

    def test_calling_first_short_raise_resets_personal_reopen_threshold(self):
        h = HoldemHand([1000, 1000, 1000, 140, 1000, 180], 5)
        h.step({"action": "raise", "amount": 100})  # seat 2
        h.step({"action": "raise", "amount": 140})  # seat 3
        h.step({"action": "call"})                  # seat 4 faces 140
        h.step({"action": "raise", "amount": 180})  # seat 5
        h.step({"action": "call"})                  # seat 0
        h.step({"action": "call"})                  # seat 1
        self.assertIn("raise", h.legal_actions())    # seat 2 faces +80
        h.step({"action": "call"})
        self.assertEqual(h.to_act, 4)
        self.assertEqual(h.observation()["to_call"], 40)
        self.assertNotIn("raise", h.legal_actions()) # seat 4 faces only +40

    def test_checker_cannot_raise_incomplete_opening_allin(self):
        h = HoldemHand([100, 100, 25, 100], 0)
        for _ in range(3):
            h.step({"action": "call"})
        h.step({"action": "check"})
        self.assertEqual((h.street, h.to_act), ("flop", 1))
        h.step({"action": "check"})
        h.step({"action": "raise", "amount": 5})
        self.assertEqual(h.observation()["min_raise_to"], 25)
        self.assertIn("raise", h.legal_actions()) # unacted seat 3
        h.step({"action": "call"})
        h.step({"action": "call"})
        self.assertEqual(h.to_act, 1)
        self.assertNotIn("raise", h.legal_actions())

    def test_short_blind_keeps_nominal_bringin_multiway(self):
        h = HoldemHand([100, 100, 5], 0)
        self.assertEqual(h.to_act, 0)
        self.assertEqual(h.observation()["to_call"], 20)
        self.assertEqual(h.observation()["min_raise_to"], 40)
        h.step({"action": "call"})
        h.step({"action": "call"})
        self.assertEqual((h.street, h.pot), ("flop", 45))

    def test_short_heads_up_blind_no_uncontestable_action(self):
        h = HoldemHand([100, 5], 0)
        self.assertTrue(h.done)
        self.assertEqual(sum(h.stacks), 105)
        self.assertEqual(h.contributions, [5, 5])
        self.assertEqual(len(h.board), 5)

    def test_one_allin_opponent_disallows_raise(self):
        h = HoldemHand([100, 15], 0)
        self.assertEqual(h.to_act, 0)
        self.assertEqual(h.observation()["to_call"], 5)
        self.assertEqual(h.legal_actions(), ["fold", "call"])
        h.step({"action": "call"})
        self.assertTrue(h.done)
        self.assertEqual(h.contributions, [15, 15])

    def test_uncalled_raise_is_returned(self):
        holes = [cards("Ac Ad"), cards("Kc Kd")]
        board = cards("2c 3d 5h 7s 9c")
        h = HoldemHand([100, 30], 0, deck=fixed_deck(holes, board))
        h.step({"action": "raise", "amount": 100})
        h.step({"action": "call"})
        self.assertTrue(h.done)
        self.assertEqual(h.contributions, [30, 30])
        self.assertEqual(h.stacks, [130, 0])
        self.assertEqual(h.awards, [60, 0])
        self.assertEqual(sum(e["paid"] for e in h.history if e["action"] == "refund"), -70)

    def test_sidepots_three_distinct_stacks(self):
        # Short stack has aces; medium kings; deep queens.
        holes = [cards("Ac Ad"), cards("Kc Kd"), cards("Qc Qd")]
        board = cards("2c 3d 5h 7s 9c")
        h = HoldemHand([50, 100, 200], 0, deck=fixed_deck(holes, board))
        h.step({"action": "raise", "amount": 50})
        h.step({"action": "raise", "amount": 100})
        h.step({"action": "call"})
        self.assertTrue(h.done)
        self.assertEqual(h.stacks, [150, 100, 100])
        self.assertEqual([p["amount"] for p in h.side_pots], [150, 100])
        self.assertEqual([p["winners"] for p in h.side_pots], [[0], [1]])

    def test_folded_contributions_stay_in_pots(self):
        holes = [cards("Ac Ad"), cards("Kc Kd"), cards("Qc Qd")]
        board = cards("2c 3d 5h 7s 9c")
        h = HoldemHand([50, 200, 200], 0, deck=fixed_deck(holes, board))
        h.step({"action": "raise", "amount": 50})
        h.step({"action": "call"})
        h.step({"action": "raise", "amount": 150})
        h.step({"action": "fold"})
        self.assertTrue(h.done)
        self.assertEqual(h.contributions, [50, 50, 50])
        self.assertEqual(h.stacks, [150, 150, 150])

    def test_tied_sidepot_while_short_stack_wins_main(self):
        holes = [cards("Ac Ad"), cards("Kc Kd"), cards("Kh Ks")]
        board = cards("2c 3d 5h 7s 9c")
        h = HoldemHand([50, 100, 200], 0, deck=fixed_deck(holes, board))
        h.step({"action": "raise", "amount": 50})
        h.step({"action": "raise", "amount": 100})
        h.step({"action": "call"})
        self.assertEqual(h.stacks, [150, 50, 150])
        self.assertEqual(h.side_pots[0]["winners"], [0])
        self.assertEqual(h.side_pots[1]["winners"], [1, 2])

    def test_board_plays_and_odd_chip_clockwise(self):
        # Player 0 folds 1-chip SB, players 1 and 2 chop a royal-flush board.
        holes = [cards("2c 3d"), cards("4c 5d"), cards("6c 7d")]
        board = cards("Th Jh Qh Kh Ah")
        h = HoldemHand([100] * 3, 2, 1, 2, deck=fixed_deck(holes, board, button=2))
        h.step({"action": "call"})  # button 2
        h.step({"action": "fold"})  # SB 0
        h.step({"action": "check"}) # BB 1
        call_check(h)
        self.assertEqual(h.stacks, [99, 101, 100])
        self.assertEqual(h.awards, [0, 3, 2])

    def test_full_board_tie_at_allin(self):
        holes = [cards("2c 3d"), cards("4c 5d")]
        board = cards("Th Jh Qh Kh Ah")
        h = HoldemHand([20, 20], 0, deck=fixed_deck(holes, board))
        h.step({"action": "call"})
        self.assertEqual(h.stacks, [20, 20])

    def test_no_hole_or_future_board_leak(self):
        h = HoldemHand([2000] * 8, 0)
        obs = h.observation()
        self.assertEqual(len(obs["hole"]), 2)
        self.assertEqual(obs["board"], [])
        self.assertNotIn("holes", obs)
        self.assertNotIn("deck", obs)
        obs["stacks"][0] = -999
        obs["history"][0]["paid"] = -999
        self.assertGreater(h.stacks[0], 0)
        self.assertGreater(h.history[0]["paid"], 0)

    def test_fallbacks_and_timeout_enforcement(self):
        def broken(obs):
            return {"action": "raise", "amount": -2}
        h = HoldemHand([20] * 3, 0)
        result = h.play([broken, CallingStation(), CallingStation()])
        self.assertEqual(result["invalid_actions"][0], 1)
        self.assertEqual(sum(result["stacks"]), 60)
        h = HoldemHand([20, 20], 0)
        result = h.play([CallingStation(), CallingStation()], decision_limit_ns=0)
        self.assertEqual(result["timeouts"][0], 1)
        self.assertEqual(result["stacks"], [10, 30])

    def test_random_legal_fuzz_and_conservation(self):
        for seed in range(150):
            rng = random.Random(seed)
            n = rng.randint(2, 8)
            stacks = [rng.randint(1, 200) for _ in range(n)]
            h = HoldemHand(stacks, rng.randrange(n), 5, 10, rng=rng)
            result = h.play([RandomLegal(seed * 8 + i) for i in range(n)])
            self.assertEqual(sum(result["stacks"]), sum(stacks))
            self.assertEqual(sum(result["invalid_actions"]), 0)
            self.assertEqual(sum(result["exceptions"]), 0)


class TournamentTests(unittest.TestCase):
    def test_icm_winner_take_all_equals_chip_fraction(self):
        self.assertEqual(icm_values([100, 300, 0], [1, 0]), [.25, .75, 0])

    def test_icm_two_players_direct_formula(self):
        values = icm_values([100, 300], [.7, .3])
        self.assertAlmostEqual(values[0], .25 * .7 + .75 * .3)
        self.assertAlmostEqual(values[1], .75 * .7 + .25 * .3)

    def test_icm_equal_stacks_and_shift(self):
        self.assertTrue(all(abs(x - .125) < 1e-12 for x in icm_values([1] * 8, [.5, .3, .2])))
        base = icm_values([10, 30, 60], [.6, .3, .1])
        shifted = icm_values([10, 30, 60], [1.6, 1.3, 1.1])
        for a, b in zip(base, shifted):
            self.assertAlmostEqual(b - a, 1.)

    def test_icm_three_player_independent_probability_tree(self):
        result = icm_values([1, 2, 3], [3, 2, 1])
        for observed, expected in zip(result, [19 / 12, 31 / 15, 47 / 20]):
            self.assertAlmostEqual(observed, expected)

    def test_cap_icm_conserves_known_local_prizes(self):
        result = run_tournament([CallingStation()] * 8, seed=6, max_hands=0,
                                payouts=(.6, .25, .15))
        self.assertTrue(result["capped"])
        self.assertEqual(result["hands"], 0)
        self.assertTrue(all(abs(p - .125) < 1e-12 for p in result["points"]))
        self.assertTrue(all(abs(r - 4.5) < 1e-12 for r in result["placements"]))

    def test_seed_reproduction_and_bounded_game(self):
        def run():
            return run_tournament([RandomLegal(i + 9) for i in range(8)], seed=17,
                                   starting_stack=200, max_hands=40,
                                   blind_schedule=((5, 10), (10, 20)), hands_per_level=20)
        a, b = run(), run()
        for key in ("stacks", "points", "placements", "hands", "eliminated"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(sum(a["stacks"]), 1600)
        self.assertAlmostEqual(sum(a["points"]), 1.)
        self.assertLessEqual(a["hands"], 40)
        self.assertEqual(sum(a["invalid_actions"]), 0)

    def test_heads_up_transition_advances_previous_big_blind(self):
        result = run_tournament([ShoveBot(0), AlwaysFold(), CallingStation()], seed=9,
                                starting_stack=50, max_hands=2, record_hands=True)
        self.assertEqual(result["hand_logs"][0]["stacks"], [0, 40, 110])
        first, second = (log["history"] for log in result["hand_logs"])
        self.assertEqual(first[1]["seat"], 2)
        self.assertEqual((second[0]["seat"], second[1]["seat"]), (2, 1))

    def test_many_tournaments_score_conservation(self):
        for seed in range(15):
            result = run_tournament([RandomLegal(seed * 8 + i) for i in range(8)], seed=seed,
                                    starting_stack=100, max_hands=30,
                                    blind_schedule=((5, 10), (10, 20), (20, 40)), hands_per_level=10,
                                    payouts=(5, 3, 2, 1))
            self.assertAlmostEqual(sum(result["points"]), 11.)
            self.assertEqual(sum(result["stacks"]), 800)
            self.assertTrue(all(p is not None and 1 <= p <= 8 for p in result["placements"]))


if __name__ == "__main__":
    unittest.main()
