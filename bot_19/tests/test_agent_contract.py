"""Integration/semantic checks; these do not assert optimal poker strategy."""

import itertools
import random
import unittest
from unittest.mock import patch

from agent import Agent
from icm import call_threshold, hero_icm
from sim import HandState, Tournament, icm_equities


def passive(hand):
    hand.apply({"action": "call" if hand.observation()["to_call"] else "check"}, strict=True)


class MixedOpponent:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, obs):
        action = self.rng.choice(obs["legal_actions"])
        if action == "raise":
            lo, hi = obs["min_raise_to"], obs["max_raise_to"]
            amount = hi if hi < lo or self.rng.random() < .35 else self.rng.randint(lo, hi)
            return {"action": action, "amount": amount}
        return {"action": action}


class AgentContractTests(unittest.TestCase):
    def test_constructor_seed_survives_table_reset(self):
        a = Agent(seed=617)
        expected = random.Random(617).random()
        Tournament([a, Agent(seed=900)], max_hands=0).run()
        self.assertEqual(a.rng.random(), expected)
        a.reset(82)
        a.rng.random()
        a.reset()
        self.assertEqual(a.rng.random(), random.Random(82).random())

    def test_history_deduplication_and_table_reset(self):
        h = HandState([1000] * 4)
        passive(h)
        obs = h.observation()
        a = Agent()
        a.act(obs)
        profiles = [row[:] for row in a.profiles]
        a.act(obs)
        self.assertEqual(a.profiles, profiles)
        self.assertEqual(sum(p[0] for p in a.profiles), 1)
        self.assertEqual(sum(p[1] for p in a.profiles), 1)
        a.reset()
        self.assertEqual(a.profiles, [[0.0] * 6 for _ in range(8)])
        self.assertEqual(a.history_seen, 0)
        self.assertEqual(a.fallbacks, 0)

    def test_open_then_call_preserves_raise_evidence(self):
        h = HandState([1000] * 3, button=0)
        h.apply({"action": "raise", "amount": 60}, strict=True)
        h.apply({"action": "raise", "amount": 180}, strict=True)
        h.apply({"action": "fold"}, strict=True)
        h.apply({"action": "call"}, strict=True)
        obs = h.observation()
        self.assertEqual(obs["hero"], 1)
        a = Agent(params={"adapt": False})
        a._learn(obs)
        self.assertEqual(a.pre_actions[0]["action"], "call")
        self.assertIn(0, a.hand_raised)
        width = a._ranges(obs, [0])[0][0]
        self.assertLess(width, .18)

    def test_small_blind_opening_position_is_explicit(self):
        h = HandState([1000] * 4, button=0)
        h.apply({"action": "fold"}, strict=True)  # UTG 3
        h.apply({"action": "fold"}, strict=True)  # button 0
        obs = h.observation()
        self.assertEqual(obs["hero"], obs["small_blind_seat"])
        a = Agent(params={"adapt": False})
        # A middling percentile distinguishes the specified SB branch from an
        # erroneous one-player-before-button count; this checks code semantics.
        with patch("agent.preflop_percentile", return_value=.50):
            action = a.act(obs)
        self.assertEqual(action["action"], "raise")
        self.assertTrue(h._valid_action(action, h.actor))

    def test_sidepot_call_prices_only_eligible_chips(self):
        h = HandState([50, 100, 200], button=0)
        passive(h)
        h.apply({"action": "raise", "amount": 100}, strict=True)
        h.apply({"action": "raise", "amount": 200}, strict=True)
        obs = h.observation()
        self.assertEqual(obs["hero"], 0)
        captured = {}
        def capture(o, villains, widths, folds, call, pot, odds):
            captured.update(call=call, pot=pot, odds=odds)
            return "call", 0
        a = Agent()
        with patch.object(a, "_preflop", side_effect=capture):
            action = a.act(obs)
        self.assertEqual(captured["call"], 30)
        # Hero can contest50 from each opponent + own20 already committed,
        # and adds30: final main pot150, immediate cost30, pot odds20%.
        self.assertEqual(captured["pot"], 120)
        self.assertAlmostEqual(captured["odds"], .20)
        self.assertEqual(action["action"], "call")
        self.assertTrue(h._valid_action(action, h.actor))

    def test_no_wager_preserves_existing_pot(self):
        h = HandState([100] * 3)
        while h.street == "preflop":
            passive(h)
        obs = h.observation()
        captured = {}
        def capture(o, villains, widths, tilts, folds, call, pot, odds):
            captured.update(call=call, pot=pot, odds=odds)
            return "check", 0
        a = Agent()
        with patch.object(a, "_postflop", side_effect=capture):
            action = a.act(obs)
        self.assertEqual(captured, {"call": 0, "pot": 60, "odds": 0.0})
        self.assertEqual(action["action"], "check")

    def test_action_depends_only_on_public_observation(self):
        for street in ("preflop", "flop", "turn", "river"):
            h = HandState([1000] * 4, seed=144)
            while h.street != street:
                passive(h)
            obs = h.observation()
            hero = h.actor
            altered = h.deck[:]
            # Keep hero's hole cards and revealed board fixed, shuffle every
            # hidden card (opponent holes, burns and future deck) independently.
            known = set(h.holes[hero] + h.board)
            indexes = [i for i, card in enumerate(altered) if card not in known]
            values = [altered[i] for i in indexes]
            random.Random(918).shuffle(values)
            for index, card in zip(indexes, values):
                altered[index] = card
            other = HandState([1000] * 4, deck=altered)
            while other.street != street:
                passive(other)
            self.assertEqual(obs, other.observation())
            self.assertEqual(Agent(seed=773).act(obs), Agent(seed=773).act(other.observation()))

    def test_decisions_do_not_open_files_or_sockets(self):
        a = Agent()
        h = HandState([1000] * 4)
        observations = []
        for street in ("preflop", "flop", "turn", "river"):
            while h.street != street:
                passive(h)
            observations.append(h.observation())
        with patch("builtins.open", side_effect=AssertionError("gameplay file IO")), patch("socket.socket", side_effect=AssertionError("gameplay network")):
            for obs in observations:
                action = a.act(obs)
                self.assertIn(action["action"], obs["legal_actions"])
        self.assertEqual(a.fallbacks, 0)

    def test_random_adversarial_states_always_return_legal_actions(self):
        rng = random.Random(911812)
        decisions = 0
        for trial in range(180):
            n = rng.randint(2, 8)
            h = HandState([rng.randint(1, 2000) for _ in range(n)], button=rng.randrange(n), seed=trial)
            agents = [Agent(seed=trial * 11 + i) if i % 2 == 0 else MixedOpponent(trial * 11 + i) for i in range(n)]
            while not h.done:
                obs = h.observation()
                actor = agents[h.actor]
                action = actor.act(obs)
                self.assertTrue(h._valid_action(action, h.actor), (obs, action))
                h.apply(action, strict=True)
                if isinstance(actor, Agent):
                    self.assertEqual(actor.fallbacks, 0)
                    decisions += 1
        self.assertGreater(decisions, 500)

    def test_forced_allin_blind_is_priced_before_opening_range(self):
        h = HandState([2000, 11], button=0, seed=1)
        obs = h.observation()
        self.assertEqual((obs["to_call"], obs["pot"]), (1, 21))
        self.assertEqual(obs["legal_actions"], ["fold", "call"])
        # Regardless of the opening percentile, 38% showdown equity should
        # call one chip to contest a22-chip pot with no remaining betting.
        with patch("agent.preflop_percentile", return_value=.99), patch("agent.preflop_equity", return_value=.38):
            action = Agent().act(obs)
        self.assertEqual(action["action"], "call")


def reference_push_ev(obs, villains, equity):
    """Independent explicit caller outcomes + ordinary side-pot settlement.

    This intentionally does not use the policy's bit-mask layer aggregation.
    Its caller probabilities are the policy's documented approximations; the
    test concerns conditional chip accounting, not calibrated poker strategy.
    """
    hero = obs["hero"]
    rates = []
    for seat in villains:
        effective_depth = min(obs["max_raise_to"], obs["stacks"][seat] + obs["bets"][seat]) / obs["big_blind"]
        rate = min(.75, max(.06, min(.65, max(.10, .11 + .60 / (effective_depth + 1.5))) * 1.01))
        rates.append(1.0 if obs["stacks"][seat] == 0 else rate)
    value = 0.0
    for calls in itertools.product((False, True), repeat=len(villains)):
        probability = 1.0
        contributions = obs["contributions"][:]
        contributions[hero] += obs["stacks"][hero]
        active = {hero}
        rate_by_seat = {}
        for seat, rate, calls_here in zip(villains, rates, calls):
            probability *= rate if calls_here else 1.0 - rate
            if calls_here:
                active.add(seat)
                rate_by_seat[seat] = rate
                contributions[seat] += min(obs["stacks"][seat], obs["max_raise_to"] - obs["bets"][seat])
        if not probability:
            continue
        previous = 0
        receipts = 0.0
        for level in sorted(set(contributions) - {0}):
            participants = [i for i, c in enumerate(contributions) if c >= level]
            amount = (level - previous) * len(participants)
            eligible = [i for i in participants if i in active]
            if hero in eligible:
                opposing = [i for i in eligible if i != hero]
                share = 1.0 if not opposing else equity(sum(rate_by_seat[i] for i in opposing) / len(opposing), len(opposing))
                receipts += amount * share
            previous = level
        value += probability * (receipts - obs["stacks"][hero])
    return value


class PushAccountingTests(unittest.TestCase):
    def test_allin_blind_cannot_contribute_phantom_chips(self):
        h = HandState([200, 5, 200], button=0)
        obs = h.observation()
        bot = Agent(params={"adapt": False})
        with patch("agent.preflop_equity", return_value=1.0):
            actual = bot._push_ev(obs, [1, 2], obs["hole"], obs["pot"])
        expected = reference_push_ev(obs, [1, 2], lambda width, n: 1.0)
        self.assertAlmostEqual(actual, expected)
        self.assertLessEqual(actual, 205.0)  # old pot25 + remaining opponent180

    def test_heads_up_existing_blinds_are_not_added_twice(self):
        h = HandState([200, 200], button=0)
        obs = h.observation()
        bot = Agent(params={"adapt": False})
        for share in (0.0, .25, .5, 1.0):
            with patch("agent.preflop_equity", return_value=share):
                actual = bot._push_ev(obs, [1], obs["hole"], obs["pot"])
            self.assertAlmostEqual(actual, reference_push_ev(obs, [1], lambda width, n: share))

    def test_unequal_stacks_sidepots_match_explicit_caller_reference(self):
        rng = random.Random(194077)
        checked = 0
        for case in range(90):
            n = rng.randint(2, 8)
            hand = HandState([rng.randint(1, 260) for _ in range(n)], button=rng.randrange(n), seed=case)
            for _ in range(rng.randrange(n)):
                if hand.done or hand.street != "preflop":
                    break
                obs = hand.observation()
                hand.apply({"action": "fold" if obs["to_call"] and rng.random() < .15 else "call" if obs["to_call"] else "check"}, strict=True)
            if hand.done or hand.street != "preflop" or "raise" not in hand.observation()["legal_actions"]:
                continue
            obs = hand.observation()
            villains = [i for i, folded in enumerate(obs["folded"]) if not folded and i != obs["hero"]]
            bot = Agent(params={"adapt": False})
            for equity in (lambda width, n: 0.0, lambda width, n: 1.0, lambda width, n: 1 / (n + 1), lambda width, n: (.52 + .16 * width) / (n ** .6)):
                with patch("agent.preflop_equity", side_effect=lambda hole, width, n: equity(width, n)):
                    actual = bot._push_ev(obs, villains, obs["hole"], obs["pot"])
                expected = reference_push_ev(obs, villains, equity)
                self.assertAlmostEqual(actual, expected, places=8, msg=(obs, actual, expected))
                self.assertGreaterEqual(actual, -obs["stacks"][obs["hero"]] - 1e-8)
                self.assertLessEqual(actual, obs["pot"] + sum(obs["stacks"][v] for v in villains) + 1e-8)
            checked += 1
        self.assertGreater(checked, 50)


class IcmContractTests(unittest.TestCase):
    def test_hero_icm_matches_independent_full_vector_algorithm(self):
        rng = random.Random(1518)
        for _ in range(160):
            n = rng.randint(2, 8)
            stacks = [rng.randint(1, 2000) for _ in range(n)]
            payout_count = rng.randint(1, n)
            payouts = sorted([rng.randint(1, 100) for _ in range(payout_count)], reverse=True)
            expected = icm_equities(stacks, payouts)
            for hero in range(n):
                self.assertAlmostEqual(hero_icm(stacks, hero, payouts), expected[hero], places=9)

    def test_short_payout_vector_has_explicit_zero_tail(self):
        obs = {"hero": 0, "stacks": [100, 100, 100, 100], "contributions": [0, 100, 0, 0], "to_call": 100, "payouts": [100, 50]}
        threshold = call_threshold(obs, 1, .5)
        self.assertGreater(threshold, .5)
        self.assertAlmostEqual(threshold, 19 / 33)
        obs["payouts"] = [100, 50, 0, 0]
        self.assertEqual(call_threshold(obs, 1, .5), threshold)

    def test_no_payouts_and_sidepots_bypass_icm(self):
        obs = {"hero": 0, "stacks": [100, 100, 100, 100], "contributions": [0, 100, 0, 0], "to_call": 100}
        self.assertEqual(call_threshold(obs, 1, .5), .5)
        obs["payouts"] = [100, 50]
        obs["contributions"][1] = 200
        self.assertEqual(call_threshold(obs, 1, .5), .5)
        obs["contributions"][1] = 50
        obs["to_call"] = 50
        self.assertEqual(call_threshold(obs, 1, .5), .5)

    def test_allin_threshold_matches_terminal_icm_states(self):
        rng = random.Random(671)
        for _ in range(120):
            n = rng.randint(2, 8)
            own_stack = rng.randint(1, 1000)
            own_committed = rng.randint(0, 100)
            call = rng.randint(1, own_stack)
            cap = own_committed + call
            stacks = [own_stack, 0] + [rng.randint(1, 1000) for _ in range(n - 2)]
            contributions = [own_committed, cap] + [rng.randint(0, cap) for _ in range(n - 2)]
            payouts = sorted([rng.randint(0, 100) for _ in range(n)], reverse=True)
            obs = {"hero": 0, "stacks": stacks, "contributions": contributions, "to_call": call, "payouts": payouts}
            pot = sum(contributions)
            fold = stacks[:]
            fold[1] += pot
            lose = stacks[:]
            lose[0] -= call
            lose[1] += pot + call
            win = stacks[:]
            win[0] += pot
            uf = icm_equities(fold, payouts)[0]
            ul = icm_equities(lose, payouts)[0] if lose[0] else payouts[n - 1]
            uw = icm_equities(win, payouts)[0]
            chip = call / (pot + call)
            expected = max(chip, min(1.0, (uf - ul) / (uw - ul))) if uw > ul + 1e-12 else chip
            self.assertAlmostEqual(call_threshold(obs, 1, chip), expected, places=9)

    def test_satellite_threshold_above_ninety_nine_percent(self):
        obs = {"hero": 0, "stacks": [5000, 5000, 1, 1, 1, 1, 1, 1], "contributions": [0, 5000, 0, 0, 0, 0, 0, 0], "to_call": 5000, "payouts": [1, 1, 0, 0, 0, 0, 0, 0]}
        uf = icm_equities([5000, 10000, 1, 1, 1, 1, 1, 1], obs["payouts"])[0]
        uw = icm_equities([10000, 5000, 1, 1, 1, 1, 1, 1], obs["payouts"])[0]
        expected = uf / uw  # losing is an eighth-place finish worth zero
        actual = call_threshold(obs, 1, .5)
        self.assertGreater(actual, .99)
        self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
