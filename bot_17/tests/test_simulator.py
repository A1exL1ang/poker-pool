import itertools
import math
import random
import unittest

from baselines import CallingAgent, FoldAgent, PressureAgent, RandomAgent, ShoveAgent, TightAgent
from simulator import Hand, Table, award_pots, evaluate, icm


def cards(text):
    ranks, suits = "23456789TJQKA", "cdhs"
    return [4 * ranks.index(s[0]) + suits.index(s[1]) for s in text.split()]


def reference_five(cs):
    """Independent straightforward five-card oracle used for evaluator tests."""
    rs = [c // 4 for c in cs]
    grouped = sorted(((rs.count(r), r) for r in set(rs)), reverse=True)
    unique = sorted(set(rs))
    flush = len({c % 4 for c in cs}) == 1
    is_straight = len(unique) == 5 and (unique[-1] - unique[0] == 4 or unique == [0, 1, 2, 3, 12])
    high = 3 if unique == [0, 1, 2, 3, 12] else unique[-1]
    if flush and is_straight:
        return (8, high)
    if grouped[0][0] == 4:
        return (7, grouped[0][1], grouped[1][1])
    if [x[0] for x in grouped] == [3, 2]:
        return (6, grouped[0][1], grouped[1][1])
    if flush:
        return (5, *sorted(rs, reverse=True))
    if is_straight:
        return (4, high)
    if grouped[0][0] == 3:
        return (3, grouped[0][1], *sorted([r for r in rs if r != grouped[0][1]], reverse=True))
    if [x[0] for x in grouped[:2]] == [2, 2]:
        return (2, max(grouped[0][1], grouped[1][1]), min(grouped[0][1], grouped[1][1]), grouped[2][1])
    if grouped[0][0] == 2:
        return (1, grouped[0][1], *sorted([r for r in rs if r != grouped[0][1]], reverse=True))
    return (0, *sorted(rs, reverse=True))


class EvaluatorTests(unittest.TestCase):
    def test_category_order(self):
        examples = ["Ac Jd 9h 6s 3d", "Ac Ad 9h 6s 3d", "Ac Ad 9h 9s 3d",
                    "Ac Ad Ah 6s 3d", "Ac 2d 3h 4s 5d", "Ac Jc 9c 6c 3c",
                    "Ac Ad Ah 6s 6d", "Ac Ad Ah As 3d", "Ac Kc Qc Jc Tc"]
        values = [evaluate(cards(s)) for s in examples]
        self.assertEqual([v[0] for v in values], list(range(9)))
        self.assertEqual(values, sorted(values))

    def test_best_of_seven_edge_cases(self):
        self.assertEqual(evaluate(cards("Ac Ad Ah Kc Kd Kh 2s")), (6, 12, 11))
        self.assertEqual(evaluate(cards("Ac Ad Kc Kd Qc Qd Js")), (2, 12, 11, 10))
        self.assertEqual(evaluate(cards("Ac 2c 3c 4c 5c Kd Ks")), (8, 3))
        self.assertEqual(evaluate(cards("Ac 2d 3h 4s 5d 6c 7d")), (4, 5))
        self.assertEqual(evaluate(cards("Ac Kc Jc 9c 7c 3c 2d")), (5, 12, 11, 9, 7, 5))

    def test_random_seven_against_independent_five_oracle(self):
        rng = random.Random(1781)
        for _ in range(1500):
            sample = rng.sample(range(52), 7)
            expected = max(reference_five(cs) for cs in itertools.combinations(sample, 5))
            self.assertEqual(evaluate(sample), expected)

    def test_invalid_cards(self):
        for bad in ([0] * 5, [0, 1, 2, 3], [0, 1, 2, 3, 52]):
            with self.assertRaises(ValueError):
                evaluate(bad)


class PotTests(unittest.TestCase):
    def test_three_levels_and_refund(self):
        awards, pots = award_pots([100, 200, 300], [False] * 3, {0: (3,), 1: (2,), 2: (1,)}, 0)
        self.assertEqual(awards, [300, 200, 100])
        self.assertEqual([p["amount"] for p in pots], [300, 200, 100])
        self.assertTrue(pots[-1]["refund"])

    def test_folded_chips_cannot_win(self):
        awards, _ = award_pots([100, 200, 300], [False, True, False], {0: (3,), 2: (1,)}, 0)
        self.assertEqual(awards, [300, 0, 300])

    def test_odd_chip_left_of_button(self):
        awards, _ = award_pots([5, 5, 5], [False, True, False], {0: (1,), 2: (1,)}, 0)
        self.assertEqual(awards, [7, 0, 8])
        awards, _ = award_pots([5, 5, 5], [False, True, False], {0: (1,), 2: (1,)}, 2)
        self.assertEqual(awards, [8, 0, 7])

    def test_split_each_side_pot_separately(self):
        awards, pots = award_pots([5, 5, 9, 9], [False] * 4, {0: (2,), 1: (1,), 2: (2,), 3: (2,)}, 0)
        self.assertEqual(awards, [6, 0, 11, 11])
        self.assertEqual(sum(awards), 28)
        self.assertEqual(len(pots), 2)


class ICMTests(unittest.TestCase):
    def test_heads_up(self):
        result = icm([75, 25], [70, 30])
        self.assertAlmostEqual(result[0], 60)
        self.assertAlmostEqual(result[1], 40)

    def test_equal_stacks_and_sum(self):
        for value in icm([100] * 8, [8, 7, 6, 5, 4, 3, 2, 1]):
            self.assertAlmostEqual(value, 4.5)
        result = icm([1, 3, 9, 27, 81], [50, 30, 20, 0, 0])
        self.assertAlmostEqual(sum(result), 100)
        self.assertEqual(result, sorted(result))

    def test_winner_take_all_is_chip_share(self):
        result = icm([10, 20, 70], [1, 0, 0])
        for actual, expected in zip(result, [0.1, 0.2, 0.7]):
            self.assertAlmostEqual(actual, expected)

    def test_zero_stacks_split_residual(self):
        self.assertEqual(icm([0, 100, 0], [50, 30, 20]), [25, 50, 25])
        self.assertEqual(icm([0, 0], [50, 30]), [40, 40])

    def test_against_permutation_oracle(self):
        stacks = [5, 9, 14, 20]
        payouts = [50, 30, 20, 0]
        expected = [0.0] * 4
        for order in itertools.permutations(range(4)):
            probability = 1.0
            remaining = sum(stacks)
            for i in order:
                probability *= stacks[i] / remaining
                remaining -= stacks[i]
            for place, i in enumerate(order):
                expected[i] += probability * payouts[place]
        for actual, value in zip(icm(stacks, payouts), expected):
            self.assertAlmostEqual(actual, value)


class BettingTests(unittest.TestCase):
    def test_heads_up_button_order_and_bb_option(self):
        hand = Hand([1000, 1000], button=0)
        self.assertEqual(hand.actor, 0)
        self.assertEqual(hand.observation()["to_call"], 10)
        hand.step({"action": "call"})
        self.assertEqual(hand.actor, 1)
        self.assertEqual(hand.observation()["legal_actions"], ["check", "raise"])
        hand.step({"action": "check"})
        self.assertEqual(hand.street, 1)
        self.assertEqual(hand.actor, 1)
        self.assertEqual(len(hand.board), 3)

    def test_eight_way_order(self):
        hand = Hand([1000] * 8, button=3)
        self.assertEqual((hand.sb_seat, hand.bb_seat, hand.actor), (4, 5, 6))
        order = []
        while hand.street == 0:
            order.append(hand.actor)
            hand.step({"action": "call" if hand.observation()["to_call"] else "check"})
        self.assertEqual(order, [6, 7, 0, 1, 2, 3, 4, 5])
        self.assertEqual(hand.actor, 4)

    def test_raise_is_total_contribution(self):
        hand = Hand([1000, 1000], button=0)
        hand.step({"action": "raise", "amount": 60})
        self.assertEqual(hand.players[0].stack, 940)
        self.assertEqual(hand.players[0].bet, 60)
        obs = hand.observation()
        self.assertEqual(obs["to_call"], 40)
        self.assertEqual(obs["min_raise_to"], 100)
        self.assertEqual(obs["pot"], 80)
        self.assertEqual(obs["history"][-1]["paid"], 50)

    def test_single_short_allin_does_not_reopen(self):
        hand = Hand([1000, 130, 1000], button=0)
        hand.step({"action": "raise", "amount": 100})
        hand.step({"action": "raise", "amount": 130})
        hand.step({"action": "call"})
        self.assertEqual(hand.actor, 0)
        self.assertFalse(hand.observation()["can_raise"])
        self.assertEqual(hand.observation()["min_raise_to"], 210)
        self.assertFalse(hand.valid_action({"action": "raise", "amount": 210}))

    def test_cumulative_short_allins_reopen_per_player(self):
        hand = Hand([1000, 140, 1000, 180, 1000], button=2)
        hand.step({"action": "raise", "amount": 100})
        hand.step({"action": "raise", "amount": 140})
        hand.step({"action": "call"})
        hand.step({"action": "raise", "amount": 180})
        hand.step({"action": "call"})
        self.assertEqual(hand.actor, 0)
        self.assertTrue(hand.observation()["can_raise"])
        self.assertEqual(hand.observation()["min_raise_to"], 260)
        hand.step({"action": "call"})
        self.assertEqual(hand.actor, 2)
        self.assertFalse(hand.observation()["can_raise"])

    def test_check_then_short_open_does_not_reopen(self):
        hand = Hand([1000, 25, 1000], button=2)
        for kind in ("call", "call", "check"):
            hand.step({"action": kind})
        self.assertEqual((hand.street, hand.actor), (1, 0))
        hand.step({"action": "check"})
        hand.step({"action": "raise", "amount": 5})
        self.assertEqual(hand.observation()["min_raise_to"], 25)
        hand.step({"action": "call"})
        self.assertEqual(hand.actor, 0)
        self.assertFalse(hand.observation()["can_raise"])

    def test_unacted_player_may_full_raise_after_short_open(self):
        hand = Hand([1000, 25, 1000], button=2)
        for kind in ("call", "call", "check", "check"):
            hand.step({"action": kind})
        hand.step({"action": "raise", "amount": 5})
        hand.step({"action": "raise", "amount": 25})
        self.assertEqual(hand.actor, 0)
        self.assertTrue(hand.observation()["can_raise"])
        self.assertEqual(hand.observation()["min_raise_to"], 45)

    def test_no_raise_when_only_opponent_is_all_in(self):
        hand = Hand([100, 1000], button=0)
        hand.step({"action": "raise", "amount": 100})
        self.assertFalse(hand.observation()["can_raise"])
        self.assertEqual(hand.observation()["to_call"], 80)

    def test_short_bb_still_full_price_multiway(self):
        hand = Hand([1000, 1000, 5], button=0)
        self.assertEqual(hand.observation()["to_call"], 20)
        hand.step({"action": "call"})
        self.assertEqual(hand.observation()["to_call"], 10)
        hand.step({"action": "call"})
        self.assertEqual(hand.observation()["pot"], 45)

    def test_short_bb_heads_up_refund_and_runout(self):
        hand = Hand([1000, 5], button=0)
        self.assertTrue(hand.done)
        self.assertEqual(hand.refunds, [5, 0])
        self.assertEqual(len(hand.board), 5)
        self.assertEqual(sum(p.stack for p in hand.players), 1005)

    def test_short_sb_heads_up_refund_and_runout(self):
        hand = Hand([5, 1000], button=0)
        self.assertTrue(hand.done)
        self.assertEqual(hand.refunds, [0, 15])
        self.assertEqual(sum(p.stack for p in hand.players), 1005)

    def test_short_bb_after_others_fold_no_extra_call(self):
        hand = Hand([1000, 1000, 5], button=0)
        hand.step({"action": "fold"})
        self.assertTrue(hand.done)
        self.assertEqual(hand.refunds, [0, 5, 0])

    def test_illegal_actions_rejected(self):
        hand = Hand([1000, 1000], button=0)
        for action in ({"action": "check"}, {"action": "raise", "amount": 21},
                       {"action": "raise", "amount": 40.0}, {"action": "raise", "amount": 1001},
                       {"action": "raise", "amount": True}, {"action": "bet", "amount": 40}, None):
            self.assertFalse(hand.valid_action(action))
        with self.assertRaises(ValueError):
            hand.step({"action": "check"})

    def test_observation_contains_only_public_information(self):
        hand = Hand([1000] * 8, button=0)
        obs = hand.observation()
        self.assertEqual(obs["hole"], hand.holes[obs["hero"]])
        self.assertNotIn("holes", obs)
        self.assertNotIn("deck", obs)
        self.assertEqual(obs["board"], [])
        for p in obs["players"]:
            self.assertNotIn("hole", p)
        obs["players"][0]["stack"] = 9
        obs["history"][0]["paid"] = 99
        self.assertNotEqual(hand.players[0].stack, 9)
        self.assertNotEqual(hand.history[0]["paid"], 99)

    def test_fold_awards_full_pot_without_showing_board(self):
        hand = Hand([1000, 1000], button=0)
        hand.step({"action": "fold"})
        self.assertTrue(hand.done)
        self.assertEqual([p.stack for p in hand.players], [990, 1010])
        self.assertEqual(hand.board, [])

    def test_sidepot_in_real_hand(self):
        hand = Hand([100, 200, 300], button=0, rng=random.Random(17))
        hand.step({"action": "raise", "amount": 100})
        hand.step({"action": "raise", "amount": 200})
        self.assertFalse(hand.observation()["can_raise"])
        hand.step({"action": "call"})
        self.assertTrue(hand.done)
        self.assertEqual([p["amount"] for p in hand.pots], [300, 200])
        self.assertEqual(sum(p.stack for p in hand.players), 600)

    def test_all_check_runout_has_three_burns(self):
        hand = Hand([1000, 1000], button=0, deck=list(range(52)))
        while not hand.done:
            hand.step({"action": "call" if hand.observation()["to_call"] else "check"})
        self.assertEqual(len(hand.burns), 3)
        self.assertEqual(hand.board, [5, 6, 7, 9, 11])
        self.assertEqual(len(set(hand.holes[0] + hand.holes[1] + hand.board + hand.burns)), 12)


class TournamentTests(unittest.TestCase):
    def test_timeout_and_exception_fallback(self):
        class BadAgent:
            def act(self, obs):
                raise RuntimeError("intentional failure")
        table = Table([BadAgent(), CallingAgent()], seed=10, max_hands=2, time_limit_ns=0)
        result = table.run()
        self.assertGreater(sum(result["errors"]), 0)
        self.assertGreater(sum(result["overruns"]), 0)
        self.assertEqual(sum(result["stacks"]), 4000)

    def test_invalid_action_fallback(self):
        class BadAgent:
            def act(self, obs):
                return {"action": "raise", "amount": -1}
        result = Table([BadAgent(), CallingAgent()], seed=4, max_hands=2).run()
        self.assertGreater(result["invalid_actions"][0], 0)

    def test_bounded_random_tables_conserve_chips_and_points(self):
        classes = [RandomAgent, TightAgent, PressureAgent, CallingAgent, ShoveAgent, RandomAgent, FoldAgent, TightAgent]
        payouts = [50, 30, 20, 0, 0, 0, 0, 0]
        for seed in range(10):
            table = Table([cls(seed=seed * 8 + i) for i, cls in enumerate(classes)], seed=seed,
                          max_hands=30, payouts=payouts, levels=((10, 20), (100, 200)), hands_per_level=10)
            result = table.run()
            self.assertEqual(sum(result["stacks"]), 16000)
            self.assertAlmostEqual(sum(result["scores"]), 100)
            self.assertFalse(any(result["errors"]))
            self.assertFalse(any(result["invalid_actions"]))
            self.assertLessEqual(result["hands"], 30)
            previous = None
            for hand in table.hand_results:
                if previous is not None:
                    alive = [i for i, stack in enumerate(previous["stacks"]) if stack]
                    expected_bb = min(alive, key=lambda i: (i - previous["big_blind_seat"] - 1) % 8)
                    self.assertEqual(hand["big_blind_seat"], expected_bb)
                    if len(alive) == 2:
                        self.assertNotEqual(hand["button"], expected_bb)
                        self.assertEqual(hand["button"], hand["small_blind_seat"])
                    else:
                        self.assertEqual(hand["small_blind_seat"], previous["big_blind_seat"])
                previous = hand

    def test_same_seed_reproduces_results(self):
        def run():
            result = Table([RandomAgent(i) for i in range(4)], seed=400, max_hands=20, payouts=[4, 3, 2, 1]).run()
            result.pop("observed_over_1ms")  # Scheduling can affect measured wall time.
            return result
        self.assertEqual(run(), run())

    def test_table_resets_agent_state(self):
        class CountAgent(CallingAgent):
            def reset(self, seed=None):
                super().reset(seed)
                self.calls = 0
            def act(self, obs):
                self.calls += 1
                return super().act(obs)
        a = CountAgent()
        a.calls = 999
        Table([a, CallingAgent()], max_hands=0)
        self.assertEqual(a.calls, 0)

    def test_fuzz_random_legal_hands(self):
        rng = random.Random(170017)
        for hand_id in range(3000):
            seats = rng.randrange(2, 9)
            active = rng.sample(range(seats), rng.randrange(2, seats + 1))
            stacks = [rng.randrange(1, 500) if i in active else 0 for i in range(seats)]
            bb = rng.choice((2, 20, 50, 100))
            hand = Hand(stacks, button=rng.randrange(seats), big_blind=bb, small_blind=bb // 2,
                        rng=rng, hand_id=hand_id)
            steps = 0
            while not hand.done:
                obs = hand.observation()
                self.assertEqual(obs["pot"], sum(p["committed"] for p in obs["players"]))
                kind = rng.choice(obs["legal_actions"])
                action = {"action": kind}
                if kind == "raise":
                    lower = min(obs["min_raise_to"], obs["max_raise_to"])
                    action["amount"] = rng.choice((lower, obs["max_raise_to"], rng.randint(lower, obs["max_raise_to"])))
                self.assertTrue(hand.valid_action(action))
                hand.step(action)
                steps += 1
                self.assertLess(steps, 1000)
            self.assertEqual(sum(p.stack for p in hand.players), sum(stacks))

    def test_dead_button_with_multiple_empty_seats(self):
        # Prior BB was seat 2 and busted; next BB is 5, with no small blind.
        hand = Hand([1000, 1000, 0, 0, 0, 1000, 1000, 0], button=1,
                    sb_seat=2, bb_seat=5, rng=random.Random(17))
        self.assertEqual(hand.actor, 6)
        blinds = [a for a in hand.history if a["action"] == "blind"]
        self.assertEqual([(a["seat"], a["paid"]) for a in blinds], [(5, 20)])
        while hand.street == 0 and not hand.done:
            hand.step({"action": "call" if hand.observation()["to_call"] else "check"})
        self.assertEqual(hand.actor, 5)
        # Next hand: button advances to the dead seat 2, SB=5, BB=6.
        next_hand = Hand([1000, 1000, 0, 0, 0, 1000, 1000, 0], button=2,
                         sb_seat=5, bb_seat=6, rng=random.Random(18))
        self.assertEqual(next_hand.actor, 0)
        self.assertEqual(next_hand.sb_seat, 5)
        self.assertEqual(next_hand.bb_seat, 6)


if __name__ == "__main__":
    unittest.main()
