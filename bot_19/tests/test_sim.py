import itertools
import random
import time
import unittest
from unittest.mock import patch

from sim import HandState, Tournament, icm_equities


def cards(text):
    ranks = "23456789TJQKA"
    suits = "cdhs"
    return [ranks.index(t[0]) * 4 + suits.index(t[1]) for t in text.split()]


def fixture_deck(holes, board, button=0):
    n = len(holes)
    reserved = [c for hand in holes for c in hand] + list(board)
    if len(reserved) != len(set(reserved)):
        raise ValueError("duplicate fixture card")
    rest = [c for c in range(52) if c not in reserved]
    deal = []
    for index in range(2):
        for offset in range(1, n + 1):
            deal.append(holes[(button + offset) % n][index])
    deal += [rest.pop(0)] + list(board[:3])
    deal += [rest.pop(0), board[3], rest.pop(0), board[4]]
    return deal + rest


class CheckCall:
    def act(self, obs):
        return {"action": "call" if obs["to_call"] else "check"}


class RandomLegal:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, obs):
        action = self.rng.choice(obs["legal_actions"])
        if action == "raise":
            minimum, maximum = obs["min_raise_to"], obs["max_raise_to"]
            target = maximum if maximum < minimum else self.rng.randint(minimum, maximum)
            return {"action": action, "amount": target}
        return {"action": action}


class JamLegal:
    def act(self, obs):
        if "raise" in obs["legal_actions"]:
            return {"action": "raise", "amount": obs["max_raise_to"]}
        return {"action": "call" if obs["to_call"] else "check"}


def do(hand, action, amount=None):
    result = {"action": action}
    if amount is not None:
        result["amount"] = amount
    return hand.apply(result, strict=True)


class BettingTests(unittest.TestCase):
    def test_preflop_order_and_big_blind_option(self):
        h = HandState([100] * 3, button=0)
        self.assertEqual(h.actor, 0)
        o = h.observation()
        self.assertEqual(o["pot"], 30)
        self.assertEqual(o["to_call"], 20)
        self.assertEqual(o["min_raise_to"], 40)
        do(h, "call")
        self.assertEqual(h.actor, 1)
        do(h, "call")
        self.assertEqual(h.actor, 2)
        self.assertEqual(h.observation()["to_call"], 0)
        self.assertIn("raise", h.observation()["legal_actions"])
        do(h, "check")
        self.assertEqual(h.street, "flop")
        self.assertEqual(h.actor, 1)
        self.assertEqual(h.pot, 60)
        self.assertEqual(h.observation()["min_raise_to"], 20)

    def test_heads_up_button_is_small_blind(self):
        h = HandState([100, 100], button=1)
        self.assertEqual((h.small_blind_seat, h.big_blind_seat), (1, 0))
        self.assertEqual(h.actor, 1)
        self.assertEqual(h.holes[0][0], h.deck[0])
        self.assertEqual(h.holes[1][0], h.deck[1])
        do(h, "call")
        self.assertEqual(h.actor, 0)
        do(h, "check")
        self.assertEqual(h.street, "flop")
        self.assertEqual(h.actor, 0)

    def test_raise_to_semantics_and_call_clipping(self):
        h = HandState([100, 100, 55], button=0)
        do(h, "raise", 60)
        self.assertEqual(h.bets[0], 60)
        self.assertEqual(h.observation()["min_raise_to"], 100)
        e = do(h, "call")
        self.assertEqual((e["amount"], e["total"], e["to_call_before"], e["stack_before"]), (50, 60, 50, 90))
        self.assertEqual(h.observation()["to_call"], 40)
        e = do(h, "call")
        self.assertEqual((e["amount"], e["total"]), (35, 55))
        self.assertEqual(h.contributions, [60, 60, 55])

    def test_single_short_allin_does_not_reopen(self):
        h = HandState([1000, 130, 1000, 1000], button=1)
        self.assertEqual(h.actor, 0)
        do(h, "raise", 100)
        do(h, "raise", 130)
        self.assertEqual(h.observation()["min_raise_to"], 210)
        do(h, "call")
        do(h, "call")
        self.assertEqual(h.actor, 0)
        self.assertEqual(h.observation()["to_call"], 30)
        self.assertFalse(h.observation()["can_raise"])
        with self.assertRaises(ValueError):
            do(h, "raise", 210)
        do(h, "call")
        self.assertEqual(h.street, "flop")

    def test_cumulative_short_allins_reopen_player_by_player(self):
        h = HandState([1000, 130, 1000, 200, 1000, 1000], button=3)
        do(h, "raise", 100)
        do(h, "raise", 130)
        do(h, "call")  # seat 2 last acted at 130
        do(h, "raise", 200)
        do(h, "call")
        do(h, "call")
        self.assertEqual(h.actor, 0)
        self.assertTrue(h.observation()["can_raise"])
        self.assertEqual(h.observation()["min_raise_to"], 280)
        do(h, "call")  # seat 0 faced 100 >= last full increment 80
        self.assertEqual(h.actor, 2)
        self.assertFalse(h.observation()["can_raise"])  # seat 2 faces only 70
        do(h, "call")
        self.assertEqual(h.street, "flop")

    def test_full_raise_reopens_and_updates_minimum(self):
        h = HandState([1000] * 3, button=0)
        do(h, "raise", 100)
        do(h, "raise", 300)
        do(h, "call")
        self.assertEqual(h.actor, 0)
        self.assertTrue(h.observation()["can_raise"])
        self.assertEqual(h.observation()["min_raise_to"], 500)
        do(h, "raise", 500)
        self.assertEqual(h.observation()["min_raise_to"], 700)

    def test_short_open_after_check(self):
        h = HandState([100, 100, 30], button=0)
        do(h, "call")
        do(h, "call")
        do(h, "check")
        self.assertEqual(h.actor, 1)
        do(h, "check")
        do(h, "raise", 10)  # remaining stack below a full 20-chip opening bet
        self.assertEqual(h.actor, 0)
        self.assertTrue(h.observation()["can_raise"])
        self.assertEqual(h.observation()["min_raise_to"], 30)
        do(h, "call")
        self.assertEqual(h.actor, 1)
        self.assertFalse(h.observation()["can_raise"])

    def test_nominal_big_blind_if_post_is_short(self):
        h = HandState([100, 100, 5], button=0)
        self.assertEqual(h.observation()["to_call"], 20)
        self.assertEqual(h.observation()["min_raise_to"], 40)
        do(h, "call")
        do(h, "call")
        self.assertEqual(h.contributions, [20, 20, 5])
        self.assertEqual(h.street, "flop")

    def test_no_betting_into_only_allin_opponent(self):
        h = HandState([100, 5], button=0)
        self.assertTrue(h.done)
        self.assertEqual(h.contributions, [5, 5])
        self.assertEqual(h.refunds, [5, 0])
        self.assertEqual(len(h.board), 5)
        self.assertEqual(sum(h.stacks), 105)

    def test_raise_when_every_other_player_allin_is_illegal(self):
        h = HandState([50, 100, 200], button=0)
        do(h, "raise", 50)
        do(h, "raise", 100)
        self.assertFalse(h.observation()["can_raise"])
        self.assertEqual(h.observation()["legal_actions"], ["fold", "call"])

    def test_invalid_actions_safe_fallback_and_strict_mode(self):
        h = HandState([100] * 3)
        for malformed in ({"action": "raise", "amount": True}, {"action": "raise", "amount": 25}, {"action": "check"}, {}, None):
            with self.assertRaises(ValueError):
                h.apply(malformed, strict=True)
        h.apply({"action": "raise", "amount": 25})
        self.assertTrue(h.folded[0])
        self.assertEqual(h.stats.invalids[0], 1)
        do(h, "call")
        h.apply({"action": "fold"})  # BB can check: invalid free fold becomes check
        self.assertFalse(h.folded[2])
        self.assertEqual(h.street, "flop")


class PotAndObservationTests(unittest.TestCase):
    def test_side_pots_and_uncalled_refund(self):
        holes = [cards("Ac Ad"), cards("Kc Kd"), cards("Qc Qd")]
        board = cards("2h 3s 7c 8h 9s")
        h = HandState([50, 100, 200], button=0, deck=fixture_deck(holes, board))
        do(h, "call")
        do(h, "raise", 100)
        do(h, "raise", 200)
        do(h, "call")
        self.assertTrue(h.done)
        self.assertEqual(h.contributions, [50, 100, 100])
        self.assertEqual(h.refunds, [0, 0, 100])
        self.assertEqual(h.awards, [150, 100, 0])
        self.assertEqual(h.stacks, [150, 100, 100])

    def test_folded_chips_and_nested_side_pots(self):
        holes = [cards("Ac Ad"), cards("Jc Jd"), cards("Kc Kd"), cards("Qc Qd")]
        board = cards("2h 3s 7c 8h 9s")
        h = HandState([60, 100, 200, 200], button=1, deck=fixture_deck(holes, board, button=1))
        do(h, "call")
        do(h, "call")
        do(h, "raise", 100)
        do(h, "call")
        do(h, "call")
        do(h, "fold")
        self.assertEqual(h.street, "flop")
        do(h, "raise", 100)
        do(h, "call")
        self.assertTrue(h.done)
        self.assertEqual(h.contributions, [60, 20, 200, 200])
        self.assertEqual(h.awards, [200, 0, 280, 0])
        self.assertEqual(h.stacks, [200, 80, 280, 0])
        self.assertNotIn(1, h.result().showdown)

    def test_split_board_and_odd_chip_left_of_button(self):
        holes = [cards("2c 3d"), cards("4c 5d"), cards("6c 7d")]
        board = cards("Th Jh Qh Kh Ah")
        h = HandState([101] * 3, small_blind=1, big_blind=2, deck=fixture_deck(holes, board))
        do(h, "call")
        do(h, "fold")
        result = h.run([CheckCall()] * 3)
        self.assertEqual(result.awards, [2, 0, 3])
        self.assertEqual(result.stacks, [101, 100, 102])

    def test_fold_win_refund_and_unrevealed_cards(self):
        h = HandState([100] * 3)
        do(h, "raise", 80)
        do(h, "fold")
        do(h, "fold")
        self.assertTrue(h.done)
        self.assertEqual(h.refunds, [60, 0, 0])
        self.assertEqual(h.stacks, [130, 90, 80])
        self.assertEqual(h.result().showdown, {})
        self.assertEqual(h.board, [])

    def test_observation_is_public_and_detached(self):
        h = HandState([100] * 4, button=0)
        original = h.observation()
        self.assertEqual(original["hole"], h.holes[h.actor])
        self.assertNotIn("holes", original)
        self.assertNotIn("deck", original)
        self.assertNotIn("payouts", original)
        original["hole"][0] = -1
        original["stacks"][0] = -999
        original["history"][0]["amount"] = -999
        original["active"][0] = False
        self.assertTrue(all(0 <= c < 52 for c in h.holes[h.actor]))
        self.assertGreaterEqual(h.stacks[0], 0)
        self.assertGreater(h.history[0]["amount"], 0)
        self.assertTrue(h.active[0])

    def test_dead_button_and_dead_small_blind(self):
        h = HandState([100, 0, 0, 100, 100], button=1, small_blind_seat=2, big_blind_seat=3)
        self.assertEqual(h.bets, [0, 0, 0, 20, 0])
        self.assertEqual(h.actor, 4)
        self.assertEqual(h.holes[3][0], h.deck[0])
        self.assertEqual(sum(len(x) for x in h.holes), 6)

    def test_random_chip_conservation_and_legal_observations(self):
        rng = random.Random(918234)
        for trial in range(350):
            n = rng.randint(2, 8)
            stacks = [rng.randint(1, 1000) for _ in range(n)]
            h = HandState(stacks, button=rng.randrange(n), seed=trial)
            agents = [RandomLegal(trial * 11 + i) for i in range(n)]
            decisions = 0
            while not h.done:
                obs = h.observation()
                self.assertGreaterEqual(obs["to_call"], 0)
                self.assertEqual(obs["pot"], sum(obs["contributions"]))
                self.assertEqual(obs["max_raise_to"], obs["stacks"][h.actor] + obs["bets"][h.actor])
                action = agents[h.actor].act(obs)
                self.assertTrue(h._valid_action(action, h.actor))
                h.apply(action, strict=True)
                decisions += 1
                self.assertLess(decisions, 1000)
            self.assertEqual(sum(h.stacks), sum(stacks))
            self.assertTrue(all(s >= 0 for s in h.stacks))
            self.assertEqual(sum(h.awards), sum(h.contributions))


class TournamentTests(unittest.TestCase):
    def test_icm_two_player_analytic_and_equal_stacks(self):
        self.assertEqual(icm_equities([1, 3], [100, 40]), [55.0, 85.0])
        for value in icm_equities([100, 100, 100], [5, 3, 1]):
            self.assertAlmostEqual(value, 3.0)
        self.assertEqual(icm_equities([0, 3], [10]), [0.0, 10.0])

    def test_icm_matches_independent_permutation_reference(self):
        stacks, payouts = [20, 50, 100, 30, 10], [100, 50, 20, 10, 5]
        expected = [0.0] * len(stacks)
        for order in itertools.permutations(range(len(stacks))):
            p, remaining = 1.0, sum(stacks)
            for i in order:
                p *= stacks[i] / remaining
                remaining -= stacks[i]
            for rank, i in enumerate(order):
                expected[i] += p * payouts[rank]
        for actual, target in zip(icm_equities(stacks, payouts), expected):
            self.assertAlmostEqual(actual, target)

    def test_no_invented_payouts_and_hand_cap(self):
        result = Tournament([CheckCall()] * 3, seed=8, max_hands=2).run()
        self.assertIsNone(result.scores)
        self.assertEqual(result.hands, 2)
        self.assertAlmostEqual(sum(result.chip_shares), 1.0)
        self.assertEqual(sum(result.stacks), 6000)

    def test_supplied_payouts_conserved_at_cap(self):
        result = Tournament([CheckCall()] * 3, seed=8, max_hands=2, payouts=[10, 5, 2]).run()
        self.assertAlmostEqual(sum(result.scores), 17.0)
        self.assertIsNone(result.champion)
        self.assertEqual(result.ranks, [None, None, None])

    def test_reproducibility_and_blind_rotation(self):
        def play():
            return Tournament(
                [RandomLegal(90 + i) for i in range(8)], seed=31415,
                starting_stack=100, max_hands=50, record_history=True, payouts=[10, 5, 3, 1],
            ).run()
        a, b = play(), play()
        self.assertEqual(a.stacks, b.stacks)
        self.assertEqual(a.scores, b.scores)
        self.assertEqual(a.hands, b.hands)
        self.assertEqual([h.history for h in a.hand_results], [h.history for h in b.hand_results])
        self.assertAlmostEqual(sum(a.scores), 19.0)
        for before, after in zip(a.hand_results, a.hand_results[1:]):
            living = [i for i, stack in enumerate(before.stacks) if stack > 0]
            expected_bb = next((before.big_blind_seat + d) % 8 for d in range(1, 9) if (before.big_blind_seat + d) % 8 in living)
            self.assertEqual(after.big_blind_seat, expected_bb)
            if len(living) == 2:
                self.assertEqual(after.button, after.small_blind_seat)
                self.assertNotEqual(after.button, after.big_blind_seat)
            else:
                self.assertEqual(after.button, before.small_blind_seat)
                self.assertEqual(after.small_blind_seat, before.big_blind_seat)

    def test_blind_level_change(self):
        seen = []
        Tournament(
            [CheckCall()] * 2, seed=11, max_hands=3, hands_per_level=1,
            blind_schedule=[(1, 2), (2, 4), (3, 6)],
            observer=lambda o, a, t: seen.append((o["hand_id"], o["big_blind"])),
        ).run()
        self.assertEqual(set(seen), {(0, 2), (1, 4), (2, 6)})

    def test_errors_and_timeout_fallbacks_measured(self):
        class ErrorAgent:
            def act(self, obs):
                raise RuntimeError("intentional test error")
        h = HandState([100] * 3)
        result = h.run([ErrorAgent(), CheckCall(), CheckCall()])
        self.assertEqual(result.stats.errors[0], 1)
        self.assertEqual(result.history[2]["action"], "fold")

        class SlowAgent:
            def act(self, obs):
                until = time.perf_counter_ns() + 2_000_000
                while time.perf_counter_ns() < until:
                    pass
                return {"action": "raise", "amount": obs["max_raise_to"]}
        h = HandState([100] * 3)
        result = h.run([SlowAgent(), CheckCall(), CheckCall()], enforce_time_limit=True)
        self.assertEqual(result.stats.overruns[0], 1)
        self.assertEqual(result.history[2]["action"], "fold")
        self.assertGreater(result.stats.latencies_ns[0][0], 1_000_000)

    def test_reset_at_table_start(self):
        class ResetAgent(CheckCall):
            resets = 0
            def reset(self):
                self.resets += 1
        agents = [ResetAgent(), ResetAgent()]
        Tournament(agents, max_hands=0).run()
        self.assertEqual([a.resets for a in agents], [1, 1])

    def test_simultaneous_elimination_order_by_starting_stack(self):
        holes = [cards("Ac Ad"), cards("Kc Kd"), cards("Qc Qd")]
        board = cards("2h 3s Qh 8h 9s")
        deck = fixture_deck(holes, board)
        def hand_factory(*args, **kwargs):
            return HandState(*args, deck=deck, **kwargs)
        with patch("sim.HandState", side_effect=hand_factory):
            result = Tournament([JamLegal()] * 3, stacks=[50, 100, 200], button=0, max_hands=1, payouts=[50, 20, 10]).run()
        self.assertEqual(result.stacks, [0, 0, 350])
        self.assertEqual(result.ranks, [3.0, 2.0, 1.0])
        self.assertEqual(result.scores, [10.0, 20.0, 50.0])

    def test_simultaneous_equal_stack_eliminations_split_places(self):
        holes = [cards("Ac Ad"), cards("Kc Kd"), cards("Qc Qd")]
        board = cards("2h 3s Qh 8h 9s")
        deck = fixture_deck(holes, board)
        def hand_factory(*args, **kwargs):
            return HandState(*args, deck=deck, **kwargs)
        with patch("sim.HandState", side_effect=hand_factory):
            result = Tournament([JamLegal()] * 3, stacks=[100, 100, 200], button=0, max_hands=1, payouts=[50, 20, 10]).run()
        self.assertEqual(result.stacks, [0, 0, 400])
        self.assertEqual(result.ranks, [2.5, 2.5, 1.0])
        self.assertEqual(result.scores, [15.0, 15.0, 50.0])

    def test_random_tournament_conservation_and_blind_rotation(self):
        saw_dead_button = saw_dead_small_blind = saw_heads_up = False
        for seed in range(80):
            result = Tournament(
                [RandomLegal(seed * 17 + i) for i in range(8)], seed=seed + 999,
                starting_stack=200, max_hands=100, payouts=[10, 6, 4, 2, 1], record_history=True,
            ).run()
            self.assertEqual(sum(result.stacks), 1600)
            self.assertAlmostEqual(sum(result.scores), 23.0)
            self.assertEqual(sum(result.stats.invalids), 0)
            self.assertEqual(sum(result.stats.errors), 0)
            self.assertLessEqual(result.hands, 100)
            for before, after in zip(result.hand_results, result.hand_results[1:]):
                living = [i for i, stack in enumerate(before.stacks) if stack > 0]
                saw_dead_button |= before.stacks[after.button] == 0
                saw_dead_small_blind |= before.stacks[after.small_blind_seat] == 0
                saw_heads_up |= len(living) == 2
                expected_bb = next((before.big_blind_seat + d) % 8 for d in range(1, 9) if (before.big_blind_seat + d) % 8 in living)
                self.assertEqual(after.big_blind_seat, expected_bb)
                if len(living) == 2:
                    self.assertEqual(after.button, after.small_blind_seat)
                    self.assertNotEqual(after.big_blind_seat, before.big_blind_seat)
        self.assertTrue(saw_dead_button)
        self.assertTrue(saw_dead_small_blind)
        self.assertTrue(saw_heads_up)


if __name__ == "__main__":
    unittest.main()
