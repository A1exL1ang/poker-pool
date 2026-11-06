import random
import time
import unittest

from pokerbot.evaluator import card_id
from pokerbot.icm import icm_equities
from pokerbot.interface import call, is_legal, passive, raise_to, validate_observation
from simulator.baselines import BASELINES, CheckFold, RandomLegal
from simulator.engine import Hand, IllegalAction, play_hand, play_tournament
from simulator.scenarios import BLINDS, PAYOUTS


def c(text):
    return [card_id(x) for x in text.split()]


def rigged_deck(holes, board, button=0):
    """Interleaved live-seat dealing followed by burn/flop/burn/turn/burn/river."""
    n = len(holes)
    active = [(button + step) % n for step in range(1, n + 1) if holes[(button + step) % n]]
    reserved = [x for hand in holes for x in hand] + board
    if len(set(reserved)) != len(reserved):
        raise ValueError("duplicate scripted card")
    unused = [x for x in range(52) if x not in reserved]
    prefix = [holes[i][round_] for round_ in range(2) for i in active]
    prefix += [unused[0]] + board[:3] + [unused[1], board[3], unused[2], board[4]]
    return prefix + [x for x in range(52) if x not in prefix]


def act(hand, kind, amount=None):
    obs = hand.observation()
    if kind == "call":
        action = call(obs)
    elif kind == "raise":
        action = {"action": "raise", "amount": amount}
    else:
        action = {"action": kind, "amount": 0}
    hand.act(action)


class AlwaysAllIn:
    def act(self, obs):
        return raise_to(obs, obs["max_raise_to"]) if obs["can_raise"] else call(obs)


class MechanicsTests(unittest.TestCase):
    def test_eight_seat_initial_order_and_blinds(self):
        hand = Hand([2000] * 8, button=3)
        obs = hand.observation()
        self.assertEqual((hand.sb_seat, hand.bb_seat, hand.actor), (4, 5, 6))
        self.assertEqual(obs["pot"], 30)
        self.assertEqual(obs["to_call"], 20)
        self.assertEqual(obs["min_raise_to"], 40)
        self.assertEqual(obs["active_players"], 8)
        self.assertEqual(hand.stacks[4:6], [1990, 1980])
        validate_observation(obs)

    def test_heads_up_button_is_small_blind_and_order_changes(self):
        hand = Hand([200, 200], button=0)
        self.assertEqual((hand.sb_seat, hand.bb_seat, hand.actor), (0, 1, 0))
        act(hand, "call")
        self.assertEqual(hand.actor, 1)
        self.assertEqual(hand.observation()["to_call"], 0)
        self.assertTrue(hand.observation()["can_raise"])
        act(hand, "check")
        self.assertEqual((hand.street, hand.actor), (1, 1))
        self.assertEqual(len(hand.board), 3)
        self.assertEqual(hand.observation()["min_raise_to"], 20)

    def test_heads_up_transition_does_not_repeat_big_blind(self):
        holes = [c("Qh Qs"), c("Ah As"), c("Kh Ks")]
        hand = Hand([20, 100, 100], button=0,
                    deck=rigged_deck(holes, c("2c 4d 7h 9s Jc")))
        while not hand.done:
            hand.act(call(hand.observation()))
        self.assertEqual(hand.stacks[0], 0)
        self.assertEqual(hand.bb_seat, 2)
        self.assertEqual(hand.next_button(), 2)
        following = Hand(hand.stacks, button=hand.next_button())
        self.assertEqual(following.bb_seat, 1)

    def test_empty_seats_skipped(self):
        hand = Hand([0, 100, 0, 100, 0, 100], button=1)
        self.assertEqual((hand.sb_seat, hand.bb_seat, hand.actor), (3, 5, 1))
        self.assertEqual(hand.holes[0], [])
        act(hand, "fold")
        act(hand, "fold")
        self.assertTrue(hand.done)
        self.assertEqual(hand.stacks, [0, 100, 0, 90, 0, 110])

    def test_raise_to_accounting_and_minimum_increment(self):
        hand = Hand([1000] * 3, button=0)
        act(hand, "raise", 70)
        self.assertEqual(hand.committed, [70, 10, 20])
        self.assertEqual(hand.observation()["pot"], 100)
        self.assertEqual(hand.observation()["to_call"], 60)
        self.assertEqual(hand.observation()["min_raise_to"], 120)
        act(hand, "raise", 200)
        self.assertEqual(hand.observation()["min_raise_to"], 330)
        self.assertEqual(hand.history[0]["amount"], 70)
        self.assertEqual(hand.history[0]["pot_before"], 30)

    def test_short_allin_does_not_reopen_prior_raiser(self):
        hand = Hand([1000, 1000, 1000, 75], button=3)
        act(hand, "raise", 60)  # seat 2, full increment 40
        act(hand, "raise", 75)  # seat 3, short increment 15
        self.assertTrue(hand.observation()["can_raise"])  # unacted SB
        self.assertEqual(hand.observation()["min_raise_to"], 115)
        act(hand, "call")
        act(hand, "call")
        self.assertEqual(hand.actor, 2)
        self.assertFalse(hand.observation()["can_raise"])
        with self.assertRaises(IllegalAction):
            act(hand, "raise", 115)
        act(hand, "call")
        self.assertEqual(hand.street, 1)

    def test_cumulative_short_allins_reopen(self):
        hand = Hand([110, 1000, 1000, 75, 95], button=4)
        act(hand, "raise", 60)
        act(hand, "raise", 75)
        act(hand, "raise", 95)
        act(hand, "raise", 110)
        act(hand, "call")
        self.assertEqual(hand.actor, 2)
        self.assertTrue(hand.observation()["can_raise"])
        self.assertEqual(hand.observation()["min_raise_to"], 150)

    def test_reopening_depends_on_each_players_last_action(self):
        hand = Hand([110, 1000, 1000, 75, 1000, 95], button=5)
        act(hand, "raise", 60)  # seat 2
        act(hand, "raise", 75)  # seat 3
        act(hand, "call")       # seat 4, faced 75
        act(hand, "raise", 95)  # seat 5
        act(hand, "raise", 110) # seat 0
        act(hand, "call")       # seat 1
        self.assertTrue(hand.observation()["can_raise"])  # 110 - 60 >= 40
        act(hand, "call")
        self.assertEqual(hand.actor, 4)
        self.assertFalse(hand.observation()["can_raise"]) # 110 - 75 < 40

    def test_short_opening_allin_does_not_reopen_check(self):
        hand = Hand([1000, 1000, 25], button=0)
        act(hand, "call")
        act(hand, "call")
        act(hand, "check")
        act(hand, "check")  # seat 1 checks flop
        act(hand, "raise", 5)
        self.assertEqual(hand.actor, 0)
        self.assertTrue(hand.observation()["can_raise"])
        self.assertEqual(hand.observation()["min_raise_to"], 25)
        act(hand, "call")
        self.assertEqual(hand.actor, 1)
        self.assertFalse(hand.observation()["can_raise"])

    def test_partial_big_blind_keeps_full_bringin_multiway(self):
        hand = Hand([200, 200, 7], button=0)
        self.assertEqual(hand.observation()["to_call"], 20)
        self.assertEqual(hand.observation()["min_raise_to"], 40)
        act(hand, "call")
        act(hand, "call")
        self.assertEqual(hand.street, 1)
        self.assertEqual(hand.committed, [20, 20, 7])

    def test_only_live_stack_owes_actual_allin_amount(self):
        hand = Hand([200, 5, 7], button=0)
        self.assertEqual(hand.observation()["to_call"], 7)
        self.assertFalse(hand.observation()["can_raise"])
        act(hand, "call")
        self.assertTrue(hand.done)
        self.assertEqual(len(hand.board), 5)
        self.assertEqual(hand.committed, [7, 5, 7])

    def test_heads_up_short_blind_runs_out_without_phantom_action(self):
        hand = Hand([200, 5], button=0)
        self.assertTrue(hand.done)
        self.assertEqual(hand.refunds, [5, 0])
        self.assertEqual(hand.committed, [5, 5])
        self.assertEqual(sum(hand.stacks), 205)

    def test_invalid_actions_rejected_without_state_mutation(self):
        hand = Hand([100, 100], button=0)
        before = hand.observation()
        for action in ({"action": "check", "amount": 0}, {"action": "raise", "amount": 30},
                       {"action": "raise", "amount": 101}, {"action": "raise", "amount": 40.0},
                       {"action": "call", "amount": 20}, {"action": "wat", "amount": 0}):
            with self.assertRaises(IllegalAction):
                hand.act(action)
            self.assertEqual(hand.observation(), before)

    def test_observation_contains_no_other_holes_or_future_board_and_is_copied(self):
        hand = Hand([100] * 3)
        obs = hand.observation()
        self.assertEqual(obs["hole_cards"], hand.holes[hand.actor])
        self.assertEqual(obs["board"], [])
        self.assertNotIn("holes", obs)
        self.assertNotIn("deck", obs)
        obs["stacks"][0] = 9999
        obs["hole_cards"][0] = -1
        obs["history"].append({})
        self.assertNotEqual(hand.stacks[0], 9999)
        self.assertNotEqual(hand.holes[hand.actor][0], -1)
        self.assertEqual(hand.history, [])


class SettlementTests(unittest.TestCase):
    def test_sidepots_and_uncalled_refund(self):
        holes = [c("Qh Qs"), c("Ah As"), c("Kh Ks")]
        hand = Hand([200, 50, 100], button=0,
                    deck=rigged_deck(holes, c("2c 4d 7h 9s Jc")))
        act(hand, "raise", 200)
        act(hand, "call")
        act(hand, "call")
        self.assertTrue(hand.done)
        self.assertEqual(hand.refunds, [100, 0, 0])
        self.assertEqual(hand.stacks, [100, 150, 100])
        self.assertEqual([p["amount"] for p in hand.pots], [150, 100])
        self.assertEqual([p["winners"] for p in hand.pots], [[1], [2]])

    def test_three_pot_layers(self):
        holes = [c("Jh Js"), c("Ah As"), c("Kh Ks"), c("Qh Qs")]
        hand = Hand([500, 50, 100, 200], button=3,
                    deck=rigged_deck(holes, c("2c 4d 6h 8s Tc"), button=3))
        act(hand, "raise", 100)
        act(hand, "raise", 200)
        act(hand, "raise", 500)
        act(hand, "call")
        self.assertTrue(hand.done)
        self.assertEqual(hand.stacks, [300, 200, 150, 200])
        self.assertEqual(hand.refunds, [300, 0, 0, 0])
        self.assertEqual([p["amount"] for p in hand.pots], [200, 150, 200])

    def test_folded_best_hand_never_wins_and_dead_money_stays(self):
        holes = [c("Kh Ks"), c("Ah As"), c("Qh Qs")]
        hand = Hand([200, 200, 100], button=0,
                    deck=rigged_deck(holes, c("2c 4d 7h 9s Jc")))
        act(hand, "raise", 100)
        act(hand, "call")
        act(hand, "call")
        act(hand, "check")
        act(hand, "raise", 100)
        act(hand, "fold")
        self.assertTrue(hand.done)
        self.assertEqual(hand.stacks, [400, 100, 0])
        self.assertEqual(hand.refunds, [100, 0, 0])

    def test_odd_chip_goes_clockwise_left_of_button(self):
        holes = [c("As Kh"), c("Ad Kc"), c("Qh Jc")]
        hand = Hand([3, 3, 3], button=0, small_blind=1, big_blind=2,
                    deck=rigged_deck(holes, c("2c 3d 4h 8s 9c")))
        act(hand, "raise", 3)
        act(hand, "call")
        act(hand, "call")
        self.assertEqual(hand.stacks, [4, 5, 0])

    def test_board_plays_for_everyone(self):
        holes = [c("2c 3c"), c("4c 5c"), c("6c 7c")]
        hand = Hand([100] * 3, deck=rigged_deck(holes, c("Th Jh Qh Kh Ah")))
        play_hand(hand, [AlwaysAllIn()] * 3)
        self.assertEqual(hand.stacks, [100] * 3)

    def test_fold_win_refund(self):
        hand = Hand([100, 100], button=0)
        act(hand, "raise", 100)
        act(hand, "fold")
        self.assertEqual(hand.stacks, [120, 80])
        self.assertEqual(hand.refunds, [80, 0])
        self.assertEqual(hand.awards, [40, 0])


class SimulationTests(unittest.TestCase):
    def test_random_legal_hands_invariants_and_contract(self):
        rng = random.Random(444)
        for k in range(150):
            n = rng.randrange(2, 9)
            stacks = [rng.randrange(1, 2001) for _ in range(n)]
            hand = Hand(stacks, button=rng.randrange(n), seed=rng.randrange(10**9))
            observations = []
            metrics = play_hand(hand, [RandomLegal(k * 10 + i) for i in range(n)],
                                on_observation=lambda obs: (validate_observation(obs), observations.append(obs)))
            self.assertEqual(sum(hand.stacks), sum(stacks))
            self.assertTrue(all(m.errors == m.illegal == 0 for m in metrics))
            for obs in observations:
                for target in (-1, 0, obs["min_raise_to"], obs["max_raise_to"], 10**9):
                    self.assertTrue(is_legal(obs, raise_to(obs, target)))

    def test_all_baselines_return_legal_actions(self):
        policies = [cls(seed=i) for i, cls in enumerate(BASELINES.values())]
        result = play_tournament(policies, seed=71, max_hands=45, starting_stack=200,
                                 blind_levels=BLINDS["fast"], payouts=[1 / len(policies)] * len(policies),
                                 on_observation=validate_observation)
        self.assertAlmostEqual(sum(result["scores"]), 1.0)
        self.assertEqual(sum(result["stacks"]), 200 * len(policies))
        self.assertTrue(all(m["errors"] == m["illegal"] == 0 for m in result["metrics"]))

    def test_simultaneous_equal_stack_eliminations_split_places(self):
        result = play_tournament([AlwaysAllIn() for _ in range(3)], seed=81, max_hands=10,
                                 blind_levels=[(10, 20)], payouts=[.6, .3, .1])
        self.assertEqual(result["hands"], 1)
        self.assertEqual(sorted(result["scores"]), [.2, .2, .6])

    def test_cutoff_scores_are_icm_and_conserved(self):
        result = play_tournament([CheckFold() for _ in range(8)], seed=91, max_hands=1,
                                 blind_levels=BLINDS["slow"], payouts=PAYOUTS["top3"], button=0)
        self.assertTrue(result["cutoff"])
        self.assertEqual(result["scores"], icm_equities(result["stacks"], PAYOUTS["top3"]))
        self.assertAlmostEqual(sum(result["scores"]), 1.0)

    def test_errors_and_invalid_actions_fallback(self):
        def error(obs):
            raise RuntimeError("test error")
        hand = Hand([100, 100], button=0)
        metrics = play_hand(hand, [error, CheckFold()], strict=False)
        self.assertEqual(metrics[0].errors, 1)
        self.assertEqual(metrics[0].fallbacks, 1)
        hand = Hand([100, 100], button=0)
        metrics = play_hand(hand, [lambda obs: {"action": "check", "amount": 0}, CheckFold()], strict=False)
        self.assertEqual(metrics[0].illegal, 1)
        self.assertEqual(metrics[0].fallbacks, 1)

    def test_timeout_fallback_is_measured_and_enforced(self):
        def slow(obs):
            time.sleep(.002)
            return call(obs)
        hand = Hand([100, 100], button=0)
        metrics = play_hand(hand, [slow, CheckFold()], enforce_time=True)
        self.assertEqual(metrics[0].over_1ms, 1)
        self.assertEqual(metrics[0].fallbacks, 1)
        self.assertEqual(hand.stacks, [90, 110])

    def test_seed_reproducibility(self):
        kwargs = dict(seed=124, max_hands=40, starting_stack=400,
                      blind_levels=BLINDS["fast"], payouts=PAYOUTS["rank"])
        a = play_tournament([RandomLegal(i) for i in range(8)], **kwargs)
        b = play_tournament([RandomLegal(i) for i in range(8)], **kwargs)
        for key in ("scores", "stacks", "hands", "survivors", "winner", "cutoff"):
            self.assertEqual(a[key], b[key])


if __name__ == "__main__":
    unittest.main()
