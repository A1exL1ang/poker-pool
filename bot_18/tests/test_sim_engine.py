import random
import unittest

from sim.engine import Hand, IllegalAction, legal_action, tournament
from sim.icm import icm
from sim.reference import evaluate as reference


def cards(text):
    ranks, suits = "23456789TJQKA", "cdhs"
    return [ranks.index(s[0]) * 4 + suits.index(s[1]) for s in text.split()]


def rig(stacks, button, holes, board, **kwargs):
    """Make a full deterministic physical deck (including three burns)."""
    assigned = {i: cards(h) for i, h in holes.items()}
    board = cards(board)
    used = sum(assigned.values(), []) + board
    assert len(set(used)) == len(used)
    unused = [c for c in range(52) if c not in used]
    order = sorted(assigned, key=lambda s: (s - button - 1) % len(stacks))
    deck = [assigned[s][round_] for round_ in range(2) for s in order]
    deck += [unused.pop(0)] + board[:3]
    deck += [unused.pop(0), board[3], unused.pop(0), board[4]]
    deck += unused
    return Hand(stacks, button, deck=deck, **kwargs)


def finish_calling(hand):
    while not hand.done:
        obs = hand.observation()
        hand.apply({"action": "call" if obs["to_call"] else "check"})


class Calling:
    def act(self, obs):
        return {"action": "call" if obs["to_call"] else "check"}


class Jamming:
    def act(self, obs):
        if obs["can_raise"]:
            return {"action": "raise", "amount": obs["max_raise_to"]}
        return {"action": "call" if obs["to_call"] else "check"}


class SimulatorTests(unittest.TestCase):
    def test_heads_up_blinds_and_order(self):
        h = Hand([1000, 1000], 0, rng=random.Random(1))
        self.assertEqual((h.sb_seat, h.bb_seat, h.actor), (0, 1, 0))
        self.assertEqual(h.bets, [10, 20])
        h.apply({"action": "call"})
        self.assertEqual(h.actor, 1)
        h.apply({"action": "check"})
        self.assertEqual((h.street, h.actor), ("flop", 1))
        h.apply({"action": "check"})
        self.assertEqual(h.actor, 0)

    def test_multiway_order_and_big_blind_option(self):
        h = Hand([1000] * 8, 5, rng=random.Random(2))
        self.assertEqual((h.sb_seat, h.bb_seat, h.actor), (6, 7, 0))
        for actor in [0, 1, 2, 3, 4, 5, 6]:
            self.assertEqual(h.actor, actor)
            h.apply({"action": "call"})
        self.assertEqual(h.actor, 7)
        self.assertTrue(h.observation()["can_raise"])
        self.assertEqual(h.observation()["to_call"], 0)
        h.apply({"action": "check"})
        self.assertEqual((h.street, h.actor), ("flop", 6))

    def test_raise_to_includes_existing_bet(self):
        h = Hand([1000, 1000], 0)
        h.apply({"action": "raise", "amount": 60})
        self.assertEqual(h.stacks, [940, 980])
        self.assertEqual(h.invested, [60, 20])
        self.assertEqual(h.observation()["to_call"], 40)
        self.assertEqual(h.observation()["min_raise_to"], 100)

    def test_full_raise_reopens_action(self):
        h = Hand([1000] * 3, 0)
        h.apply({"action": "raise", "amount": 60})
        h.apply({"action": "raise", "amount": 100})
        h.apply({"action": "call"})
        self.assertEqual(h.actor, 0)
        self.assertTrue(h.observation()["can_raise"])

    def test_short_allin_does_not_reopen(self):
        h = Hand([1000, 150, 1000, 1000], 0)
        h.apply({"action": "raise", "amount": 100})  # seat 3, min increment 80
        h.apply({"action": "call"})                   # seat 0
        h.apply({"action": "raise", "amount": 150}) # seat 1, short all-in
        self.assertEqual(h.actor, 2)
        self.assertTrue(h.observation()["can_raise"])  # unacted BB
        h.apply({"action": "call"})
        self.assertEqual(h.actor, 3)
        self.assertFalse(h.observation()["can_raise"])
        with self.assertRaises(IllegalAction):
            h.apply({"action": "raise", "amount": 230})
        h.apply({"action": "call"})
        self.assertFalse(h.observation()["can_raise"])

    def test_cumulative_short_allins_reopen(self):
        h = Hand([1000, 150, 180, 1000], 0)
        h.apply({"action": "raise", "amount": 100})
        h.apply({"action": "call"})
        h.apply({"action": "raise", "amount": 150})
        h.apply({"action": "raise", "amount": 180})
        self.assertEqual(h.actor, 3)
        self.assertTrue(h.observation()["can_raise"])
        self.assertEqual(h.observation()["min_raise_to"], 260)

    def test_reopening_is_player_specific(self):
        h = Hand([1000, 180, 1000, 1000, 125], 0)
        h.apply({"action": "raise", "amount": 100})  # seat3
        h.apply({"action": "raise", "amount": 125})  # seat4 short all-in
        h.apply({"action": "call"})                    # seat0 acts at125
        h.apply({"action": "raise", "amount": 180})  # seat1 short all-in
        h.apply({"action": "call"})                    # seat2
        self.assertEqual(h.actor, 3)
        self.assertTrue(h.observation()["can_raise"])  # 180-100 >=80
        h.apply({"action": "call"})
        self.assertEqual(h.actor, 0)
        self.assertFalse(h.observation()["can_raise"]) # 180-125 <80

    def test_short_opening_does_not_reopen_checker(self):
        h = Hand([1000, 1000, 25], 0)
        h.apply({"action": "call"})
        h.apply({"action": "call"})
        h.apply({"action": "check"})
        self.assertEqual((h.street, h.actor), ("flop", 1))
        h.apply({"action": "check"})
        h.apply({"action": "raise", "amount": 5})
        self.assertTrue(h.observation()["can_raise"])
        self.assertEqual(h.observation()["min_raise_to"], 25)
        h.apply({"action": "call"})
        self.assertEqual(h.actor, 1)
        self.assertFalse(h.observation()["can_raise"])

    def test_short_allin_is_not_completion(self):
        h = Hand([1000, 1000, 25], 0)
        h.apply({"action": "call"})
        h.apply({"action": "call"})
        h.apply({"action": "check"})
        h.apply({"action": "check"})
        h.apply({"action": "raise", "amount": 5})
        with self.assertRaises(IllegalAction):
            h.apply({"action": "raise", "amount": 20})
        h.apply({"action": "raise", "amount": 25})
        self.assertTrue(h.observation()["can_raise"])

    def test_underminimum_raise_rejected_unless_allin(self):
        h = Hand([1000, 1000], 0)
        for amount in [21, 39, 20, 1001, 40.0, True]:
            with self.assertRaises(IllegalAction):
                h.apply({"action": "raise", "amount": amount})
        h.apply({"action": "raise", "amount": 40})

    def test_call_is_capped_by_stack(self):
        h = Hand([1000, 50], 0)
        h.apply({"action": "raise", "amount": 200})
        self.assertEqual(h.observation()["to_call"], 30)
        self.assertFalse(h.observation()["can_raise"])
        h.apply({"action": "call"})
        self.assertTrue(h.done)
        self.assertEqual(h.refunds[0], 150)
        self.assertEqual(h.invested, [50, 50])

    def test_short_big_blind_does_not_force_overcall_heads_up(self):
        h = Hand([1000, 5], 0)
        self.assertTrue(h.done)
        self.assertEqual(h.invested, [5, 5])
        self.assertEqual(h.refunds[0], 5)

    def test_short_big_blind_full_price_multiway(self):
        h = Hand([1000, 1000, 5], 0)
        self.assertFalse(h.done)
        self.assertEqual(h.actor, 0)
        self.assertEqual(h.observation()["to_call"], 20)
        self.assertEqual(h.observation()["min_raise_to"], 40)

    def test_no_betting_into_only_allin_opponents(self):
        h = Hand([1000, 80, 50], 0)
        h.apply({"action": "raise", "amount": 100})
        h.apply({"action": "call"})
        h.apply({"action": "call"})
        self.assertTrue(h.done)
        self.assertEqual(h.refunds[0], 20)

    def test_three_way_sidepots(self):
        h = rig([100, 200, 300], 0,
                {0: "As Ah", 1: "Ks Kh", 2: "Qs Qh"}, "2c 3d 7s 8h 9c")
        h.apply({"action": "raise", "amount": 100})
        h.apply({"action": "raise", "amount": 200})
        h.apply({"action": "call"})
        self.assertEqual(h.stacks, [300, 200, 100])
        self.assertEqual([p["amount"] for p in h.pots], [300, 200])
        self.assertEqual([p["winners"] for p in h.pots], [[0], [1]])

    def test_folded_money_remains_in_pots(self):
        h = rig([1000] * 3, 0,
                {0: "As Ah", 1: "Ks Kh", 2: "Qs Qh"}, "2c 3d 7s 8h 9c")
        h.apply({"action": "raise", "amount": 60})
        h.apply({"action": "call"})
        h.apply({"action": "call"})
        h.apply({"action": "check"})  # flop seat1
        h.apply({"action": "check"})
        h.apply({"action": "raise", "amount": 100})
        h.apply({"action": "fold"})
        h.apply({"action": "call"})
        finish_calling(h)
        self.assertEqual(h.stacks, [1220, 940, 840])

    def test_tie_odd_chip_left_of_button(self):
        h = rig([1000] * 3, 0,
                {0: "As Ah", 1: "Ks Kh", 2: "Qs Qh"}, "2c 3d 4s 5h 6c",
                small_blind=1, big_blind=2)
        h.apply({"action": "call"})
        h.apply({"action": "fold"})
        h.apply({"action": "check"})
        finish_calling(h)
        self.assertEqual(h.awards, [2, 0, 3])
        self.assertEqual(h.stacks, [1000, 999, 1001])

    def test_board_plays_and_wheel(self):
        self.assertEqual(reference(cards("Ac Kd Qh Js Tc 2s 2h")),
                         reference(cards("Ac Kd Qh Js Tc 3s 4h")))
        self.assertGreater(reference(cards("Ac 2d 3h 4s 5c")),
                           reference(cards("Kc Kd Qh Js Tc")))

    def test_deck_and_observation_information_boundary(self):
        h = Hand([1000] * 8, 0, rng=random.Random(3))
        o = h.observation()
        self.assertEqual(len(o["hole"]), 2)
        self.assertEqual(o["board"], [])
        self.assertNotIn("deck", o)
        self.assertNotIn("holes", o)
        self.assertTrue(all("hole" not in p for p in o["players"]))
        o["hole"][0] = -1
        o["players"][0]["stack"] = -1
        o["history"][0]["amount"] = -1
        self.assertNotEqual(h.holes[h.actor][0], -1)
        self.assertEqual(h.stacks[0], 1000)
        self.assertGreaterEqual(h.history[0]["amount"], 0)

    def test_check_facing_bet_rejected(self):
        h = Hand([1000, 1000], 0)
        with self.assertRaises(IllegalAction):
            h.apply({"action": "check"})

    def test_call_with_nothing_due_normalizes_check(self):
        h = Hand([1000, 1000], 0)
        h.apply({"action": "call"})
        h.apply({"action": "call"})
        self.assertEqual(h.street, "flop")
        self.assertEqual(h.history[-1]["kind"], "check")

    def test_dead_small_blind(self):
        h = Hand([1000, 0, 1000, 1000], 0, sb_seat=None, bb_seat=2)
        self.assertEqual(h.bets, [0, 0, 20, 0])
        self.assertEqual(h.actor, 3)

    def test_tournament_conserves_chips_and_points(self):
        for seed in range(10):
            result = tournament([Jamming() for _ in range(8)], seed=seed, max_hands=200,
                                payouts=[7, 6, 5, 4, 3, 2, 1, 0])
            self.assertEqual(sum(result.stacks), 16000)
            self.assertAlmostEqual(sum(result.points), 28)
            self.assertEqual(sum(result.errors + result.illegal), 0)
            self.assertEqual(sum(s > 0 for s in result.stacks), 1)

    def test_deterministic_tournament(self):
        a = tournament([Calling() for _ in range(8)], seed=23, max_hands=12)
        b = tournament([Calling() for _ in range(8)], seed=23, max_hands=12)
        self.assertEqual(a.stacks, b.stacks)
        self.assertEqual(a.points, b.points)

    def test_tournament_blind_progression_through_eliminations(self):
        from sim.engine import next_seat
        transitions = 0
        for seed in range(25):
            hands = []
            tournament([Jamming() for _ in range(8)], seed=seed, max_hands=80,
                       observer=hands.append)
            for previous, current in zip(hands, hands[1:]):
                live = {i for i, s in enumerate(current.initial_stacks) if s > 0}
                self.assertEqual(current.bb_seat, next_seat(previous.bb_seat, live, 8))
                if len(live) == 2:
                    self.assertEqual(current.sb_seat, current.button)
                    self.assertNotEqual(current.bb_seat, current.button)
                    if len(previous.seated) > 2:
                        transitions += 1
        self.assertGreater(transitions, 0)

    def test_exception_illegal_timeout_fallbacks(self):
        class Bad:
            def act(self, _):
                raise RuntimeError("test failure")
        class Invalid:
            def act(self, _):
                return {"action": "dance"}
        result = tournament([Bad(), Invalid(), Calling()], seed=7, max_hands=3,
                            enforce_timeout=False, timeout_ns=0)
        self.assertGreater(result.errors[0], 0)
        self.assertGreater(result.illegal[1], 0)
        self.assertEqual(result.timeouts, result.decisions)
        result2 = tournament([Calling(), Calling()], seed=1, max_hands=3,
                             enforce_timeout=True, timeout_ns=0)
        self.assertEqual(result2.timeouts, result2.decisions)
        self.assertEqual(sum(result2.errors + result2.illegal), 0)

    def test_randomized_legal_action_chip_conservation(self):
        rng = random.Random(82251)
        for _ in range(150):
            stacks = [rng.randint(1, 500) for _ in range(rng.randint(2, 8))]
            h = Hand(stacks, rng.randrange(len(stacks)), rng=rng)
            steps = 0
            while not h.done:
                o = h.observation()
                options = [{"action": "call" if o["to_call"] else "check"}, {"action": "fold"}]
                if o["can_raise"]:
                    options.append({"action": "raise", "amount": o["max_raise_to"]})
                    if o["max_raise_to"] >= o["min_raise_to"]:
                        options.append({"action": "raise", "amount": rng.randint(o["min_raise_to"], o["max_raise_to"])})
                h.apply(rng.choice(options))
                h.assert_invariants()
                steps += 1
                self.assertLess(steps, 1000)
            self.assertEqual(sum(h.stacks), sum(stacks))


class ICMTests(unittest.TestCase):
    def test_equal_stacks_equal_values(self):
        for value in icm([100] * 3, [50, 30, 20]):
            self.assertAlmostEqual(value, 100 / 3)

    def test_heads_up_linear(self):
        values = icm([75, 25], [100, 40])
        self.assertEqual(values, [85, 55])

    def test_winner_take_all_chip_fractions(self):
        self.assertEqual(icm([1, 2, 3], [60, 0, 0]), [10, 20, 30])

    def test_equal_payout_is_constant(self):
        values = icm([1, 19, 500, 2], [7, 7, 7, 7])
        for value in values:
            self.assertAlmostEqual(value, 7)

    def test_zero_stack_handling(self):
        self.assertEqual(icm([0, 10, 0], [100, 40, 20]), [30, 100, 30])
        self.assertEqual(icm([0, 0], [10, 2]), [6, 6])


if __name__ == "__main__":
    unittest.main()
