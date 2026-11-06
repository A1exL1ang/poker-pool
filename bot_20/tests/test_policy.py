"""Rule-critical policy/interface regression tests, independent of strength tuning."""
import copy
import random
import unittest

from pokerbot.interface import call, is_legal, raise_to, validate_observation
from simulator.engine import Hand, play_hand


def card(text):
    return "23456789TJQKA".index(text[0]) * 4 + "cdhs".index(text[1])


class ContractTests(unittest.TestCase):
    def test_raise_to_and_short_allin(self):
        hand = Hand([100, 25, 100], button=0, small_blind=5, big_blind=10)
        obs = hand.observation()
        validate_observation(obs)
        action = raise_to(obs, 30)
        self.assertEqual(action, {"action": "raise", "amount": 30})
        hand.act(action)
        obs = hand.observation()
        self.assertEqual(obs["to_call"], 25)
        self.assertEqual(call(obs), {"action": "call", "amount": 20})

    def test_explicit_contract_rejects_wrong_pot_and_cards(self):
        obs = Hand([100, 100]).observation()
        bad = copy.deepcopy(obs)
        bad["pot"] += 1
        with self.assertRaises(ValueError):
            validate_observation(bad)
        bad = copy.deepcopy(obs)
        bad["hole_cards"][1] = bad["hole_cards"][0]
        with self.assertRaises(ValueError):
            validate_observation(bad)

    def test_disabled_reopening_never_raises(self):
        obs = Hand([100, 100]).observation()
        obs["can_raise"] = False
        self.assertEqual(raise_to(obs, 100), call(obs))


class PolicyTests(unittest.TestCase):
    def setUp(self):
        from agent import Agent
        self.bot = Agent(seed=20)

    def test_guaranteed_chops(self):
        boards = ("Tc Jc Qc Kc Ac", "Tc Jd Qh Ks Ac", "2c 2d 2h 2s Ac", "Ac Ad Ah As Kc")
        for text in boards:
            board = [card(c) for c in text.split()]
            self.assertTrue(self.bot._unbeatable_board(board))
        self.assertFalse(self.bot._unbeatable_board([card(c) for c in "Tc Jc Qc Kc Ad".split()]))
        self.assertFalse(self.bot._unbeatable_board([card(c) for c in "2c 2d 2h 2s Kc".split()]))

    def test_guaranteed_chop_never_folds_to_huge_bet(self):
        obs = Hand([1000, 1000]).observation()
        obs.update(street=3, board=[card(c) for c in "Tc Jd Qh Ks Ac".split()],
                   hole_cards=[card("3c"), card("5s")], to_call=980,
                   committed=[20, 1000], street_committed=[0, 980],
                   stacks=[980, 0], hero=0, pot=1020, can_raise=False,
                   all_in=[False, True], payouts=[0.6, 0.4])
        self.assertEqual(self.bot.act(obs), {"action": "call", "amount": 980})

    def test_reset_discards_anonymous_opponent_history(self):
        initial_random = self.bot.rng.random()
        self.bot.stats[2].vpip = 100
        self.bot.last_hand = 7
        self.bot.reset("fresh")
        self.assertEqual(self.bot.stats[2].vpip, 0)
        self.assertEqual(self.bot.last_hand, -1)
        self.assertEqual(self.bot.rng.random(), initial_random)

    def test_satellite_bubble_value_raise_cannot_bypass_icm(self):
        hand = Hand([2000, 2000, 10, 10], button=3, payouts=[1 / 3, 1 / 3, 1 / 3, 0])
        while hand.actor != 0 or hand.street != 3:
            obs = hand.observation()
            if hand.actor in (2, 3):
                hand.act({"action": "fold", "amount": 0})
            else:
                hand.act(call(obs))
        hand.act(call(hand.observation()))
        hand.act(raise_to(hand.observation(), 1900))
        obs = hand.observation()
        obs["hole_cards"] = [card("Ah"), card("Kh")]
        obs["board"] = [card(c) for c in "2c 3d 4h 5s 9c".split()]
        self.assertGreater(self.bot._call_threshold(obs, 1900, [1], True), .99)
        self.assertEqual(self.bot.act(obs)["action"], "fold")

    def test_nearly_committing_preflop_call_prices_icm(self):
        hand = Hand([2000, 2000, 10, 10], button=3, payouts=[1 / 3, 1 / 3, 1 / 3, 0])
        for _ in range(2):
            hand.act({"action": "fold", "amount": 0})
        hand.act(call(hand.observation()))
        hand.act(raise_to(hand.observation(), 1900))
        obs = hand.observation()
        obs["hole_cards"] = [card("Ac"), card("As")]
        self.assertGreater(self.bot._call_threshold(obs, 1880, [1], False), .99)
        self.assertEqual(self.bot.act(obs)["action"], "fold")

    def test_icm_call_price_and_dead_money(self):
        obs = Hand([100, 100], small_blind=5, big_blind=10).observation()
        obs.update(hero=0, stacks=[90, 0], committed=[10, 100], pot=110,
                   hand_start_stacks=[100, 100], payouts=[1.0, 0.0])
        self.assertAlmostEqual(self.bot._call_threshold(obs, 90, [1], True), .45)
        obs["payouts"] = [.6, .4]
        self.assertAlmostEqual(self.bot._call_threshold(obs, 90, [1], True), .45)

    def test_sidepot_cap_is_not_counted_as_winnable(self):
        obs = Hand([100, 100, 100]).observation()
        obs.update(hero=0, committed=[20, 300, 300], stacks=[80, 0, 0], pot=620)
        self.assertEqual(self.bot._eligible_pot(obs, 80), 300)

    def test_aces_call_partial_allin_blinds(self):
        obs = Hand([200, 5, 7], button=0).observation()
        obs["hole_cards"] = [card("Ac"), card("As")]
        self.assertFalse(obs["can_raise"])
        self.assertEqual(self.bot.act(obs), call(obs))

    def test_empty_seat_gaps_do_not_change_position_ranges(self):
        compact = Hand([1000] * 4, button=3)
        sparse = Hand([1000, 0, 1000, 0, 0, 1000, 0, 1000], button=7)
        for hand in (compact, sparse):
            hand.act(raise_to(hand.observation(), 60))
            hand.act({"action": "fold", "amount": 0})
        a, b = compact.observation(), sparse.observation()
        a["hole_cards"] = b["hole_cards"] = [card("Kc"), card("9d")]
        ra = self.bot._ranges(a, self.bot._live(a))
        rb = self.bot._ranges(b, self.bot._live(b))
        self.assertAlmostEqual(ra[0], rb[0])
        first = self.bot.act(a)
        self.bot.reset()
        self.assertEqual(first, self.bot.act(b))

    def test_fuzz_complete_hands_legal_and_state_private(self):
        from agent import Agent
        from simulator.baselines import make_baseline
        rng = random.Random(731)
        for seed in range(50):
            n = 2 + seed % 7
            stacks = [rng.randint(1, 2500) for _ in range(n)]
            bots = [Agent(seed=seed + i) if i % 2 == 0 else make_baseline("maniac", seed + i) for i in range(n)]
            hand = Hand(stacks, button=seed % n, small_blind=10, big_blind=20, seed=seed)
            def verify(obs):
                validate_observation(obs)
                self.assertNotIn("holes", obs)
            play_hand(hand, bots, on_observation=verify)
            self.assertEqual(sum(stacks), sum(hand.stacks))
            for bot in bots:
                if isinstance(bot, Agent):
                    self.assertEqual(bot.errors, 0)


if __name__ == "__main__":
    unittest.main()
