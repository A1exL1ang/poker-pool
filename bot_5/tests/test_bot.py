import itertools
import json
import math
from pathlib import Path
import random
import unittest

from pokerbot.agent import Agent
from pokerbot.cards import evaluate, _evaluate_direct, parse_card, hand_index, BoardEvaluator
from pokerbot.equity import Equity, flop_key, preflop_equity
from pokerbot.icm import icm_equity, risk_premium
from pokerbot.protocol import legalize
from simulator import Table, Baseline, award_pots


def cards(text):
    return [parse_card(c) for c in text.split()]


def reference5(hand):
    """Independent, deliberately slow tuple evaluator for exactly five cards."""
    ranks = sorted((c // 4 + 2 for c in hand), reverse=True)
    counts = {r: ranks.count(r) for r in set(ranks)}
    grouped = sorted(((n, r) for r, n in counts.items()), reverse=True)
    flush = len({c % 4 for c in hand}) == 1
    unique = sorted(set(ranks))
    straight = unique[-1] if len(unique) == 5 and unique[-1] - unique[0] == 4 else 5 if unique == [2, 3, 4, 5, 14] else 0
    if straight and flush:
        return (8, straight)
    if grouped[0][0] == 4:
        return (7, grouped[0][1], grouped[1][1])
    if [x[0] for x in grouped] == [3, 2]:
        return (6, grouped[0][1], grouped[1][1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if grouped[0][0] == 3:
        return (3, grouped[0][1], *sorted((r for r in ranks if r != grouped[0][1]), reverse=True))
    if grouped[0][0] == grouped[1][0] == 2:
        return (2, max(grouped[0][1], grouped[1][1]), min(grouped[0][1], grouped[1][1]), grouped[2][1])
    if grouped[0][0] == 2:
        return (1, grouped[0][1], *sorted((r for r in ranks if r != grouped[0][1]), reverse=True))
    return (0, *ranks)


def reference_score(hand):
    value = max(reference5(c) for c in itertools.combinations(hand, 5))
    # Our compact scores use rank 1..13 rather than 2..14.
    score = value[0] << 20
    for j, rank in enumerate(value[1:]):
        score |= (rank - 1) << (16 - j * 4)
    return score


def observation(hole="As Ah", board="", call=20, stack=2000):
    players = [{"stack": 2000, "bet": 0, "committed": 0, "folded": False} for _ in range(8)]
    players[0]["stack"] = stack
    players[1].update(stack=1990, bet=10, committed=10)
    players[2].update(stack=1980, bet=20, committed=20)
    legal = {"fold": True, "call": min(call, stack)} if call else {"check": True}
    if stack > call:
        legal["raise"] = {"min": min(stack, max(40, call * 2)), "max": stack}
    return {"table_id": "test", "hand_id": 0, "hero": 0, "button": 0, "hole_cards": cards(hole), "board": cards(board), "players": players, "pot": max(30, call * 2), "big_blind": 20, "legal_actions": legal, "history": []}


class EvaluatorTests(unittest.TestCase):
    def test_known_categories_and_ordering(self):
        samples = ["As Js 9s 6s 3s", "As Ah Ad Ks Kd", "As Ah Ad Ac 2d", "9s Ts Js Qs Ks"]
        self.assertEqual([evaluate(cards(s)) >> 20 for s in samples], [5, 6, 7, 8])
        self.assertLess(evaluate(cards("As 2h 3d 4c 5s")), evaluate(cards("2s 3h 4d 5c 6s")))
        self.assertGreater(evaluate(cards("As Ah Ad Ks Kh Kd 2s")), evaluate(cards("Ks Kh Kd Qs Qh Qd As")))
        self.assertGreater(evaluate(cards("As Ah Ks Kh Qs Qh 2d")), evaluate(cards("As Ah Ks Kh Js Jh 2d")))
        self.assertEqual(evaluate(cards("As Ah Ks Kh Qs Qh 2d")), evaluate(cards("As Ah Ks Kh Js Jh Qd")))

    def test_independent_reference_5000_hands(self):
        rng = random.Random(71050)
        for j in range(5000):
            hand = rng.sample(range(52), 5 + j % 3)
            self.assertEqual(evaluate(hand), reference_score(hand), hand)
            self.assertEqual(evaluate(hand), _evaluate_direct(hand), hand)

    def test_suit_permutation_invariance(self):
        rng = random.Random(8)
        for _ in range(300):
            h = rng.sample(range(52), 7)
            p = rng.sample(range(4), 4)
            self.assertEqual(evaluate(h), evaluate([(c & ~3) | p[c & 3] for c in h]))

    def test_reused_board_evaluator(self):
        rng = random.Random(9916)
        for _ in range(3000):
            hand = rng.sample(range(52), 7)
            e = BoardEvaluator(hand[2:5])
            self.assertEqual(e.pair(*hand[:2]), evaluate(hand[:5]))
            e = e.add(hand[5])
            self.assertEqual(e.pair(*hand[:2]), evaluate(hand[:6]))
            e = e.add(hand[6])
            self.assertEqual(e.pair(*hand[:2]), evaluate(hand))

    def test_cards_reject_invalid(self):
        for value in (-1, 52, True, "1s", "Acx", None):
            with self.assertRaises(ValueError):
                parse_card(value)
        self.assertEqual(parse_card("10h"), parse_card("Th"))


class EquityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.equity = Equity()

    def test_asset_coverage_and_sorted_keys(self):
        self.assertEqual(len(self.equity.keys), 1286792)
        self.assertTrue(all(a < b for a, b in zip(self.equity.keys, self.equity.keys[1:])))
        self.assertEqual(len(self.equity.values), 4 * len(self.equity.keys))

    def test_canonicalization_and_lookups(self):
        from bisect import bisect_left
        rng = random.Random(99)
        for _ in range(3000):
            deal = rng.sample(range(52), 5)
            key = flop_key(deal[:2], deal[2:])
            self.assertEqual(self.equity.keys[bisect_left(self.equity.keys, key)], key)
            for p in ((1, 0, 3, 2), (3, 2, 0, 1)):
                other = [(c & ~3) | p[c & 3] for c in deal]
                self.assertEqual(flop_key(other[:2], other[2:]), key)

    def test_nuts_and_playing_the_board(self):
        self.assertEqual(self.equity.estimate(cards("As Ks"), cards("Qs Js Ts 2d 3c"), 1.0, 3), 1.0)
        self.assertEqual(self.equity.estimate(cards("2c 3d"), cards("As Ks Qs Js Ts"), 1.0, 2), 0.5)
        self.assertGreater(preflop_equity(cards("As Ah"), .03), .80)
        self.assertLess(preflop_equity(cards("7s 2h"), 1), .40)


class ICMTests(unittest.TestCase):
    def test_known_cases(self):
        self.assertAlmostEqual(icm_equity((100, 300), 0, (1, 0)), .25)
        self.assertAlmostEqual(icm_equity((100, 300), 0, (.7, .3)), .4)
        for i in range(8):
            self.assertAlmostEqual(icm_equity((2000,) * 8, i, (.5, .3, .2, 0, 0, 0, 0, 0)), .125)
        self.assertAlmostEqual(sum(icm_equity((100, 200, 700), i, (.5, .3, .2)) for i in range(3)), 1)
        self.assertEqual(risk_premium((100, 200, 700), 0, 2, (1, 0, 0)), 0)
        self.assertGreater(risk_premium((100, 100, 100, 100), 0, 1, (.5, .3, .2, 0)), 0)
        self.assertEqual(icm_equity((100, 900, 400), 0, (1, 1, 1)), 1)

    def test_independent_permutations(self):
        stacks = (100, 450, 210, 800, 55)
        payouts = (.50, .30, .15, .05, 0)
        reference = [0.0] * 5
        for order in itertools.permutations(range(5)):
            prob = 1.0
            total = sum(stacks)
            for i in order:
                prob *= stacks[i] / total
                total -= stacks[i]
            for rank, i in enumerate(order):
                reference[i] += prob * payouts[rank]
        for i in range(5):
            self.assertAlmostEqual(icm_equity(stacks, i, payouts), reference[i])


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = Agent(seed=5)

    def test_premiums_raise_and_free_checks(self):
        action = self.bot.act(observation())
        self.assertEqual(action["action"], "raise")
        obs = observation("7c 2d", call=0)
        obs["legal_actions"] = {"check": True}
        self.assertEqual(self.bot.act(obs), {"action": "check", "amount": 0})

    def test_bad_input_safe_fallback(self):
        obs = observation(call=0)
        obs["hole_cards"] = [0, 0]
        self.assertEqual(self.bot.act(obs)["action"], "check")
        self.assertIsNotNone(self.bot.last_error)
        obs["legal_actions"] = []
        self.assertEqual(self.bot.act(obs)["action"], "fold")

    def test_locked_public_nuts_never_fold(self):
        for board in ("As Ks Qs Js Ts", "As Kd Qc Jh Ts", "Ac Ad Ah As Kc", "2c 2d 2h 2s Ac"):
            obs = observation("7c 6d", board, call=2000)
            self.assertEqual(self.bot.act(obs)["action"], "call")
            self.assertEqual(self.bot.last_info["reason"], "unbeatable_shared_board")
        obs = observation("7c 6d", "As Ks Qs Jh Ts", call=2000)
        self.bot.act(obs)
        self.assertNotEqual(self.bot.last_info.get("reason"), "unbeatable_shared_board")

    def test_short_allin_clamping_and_no_reopen(self):
        self.assertEqual(legalize("raise", 500, {"fold": True, "call": 10, "raise": {"min": 17, "max": 17}}), {"action": "raise", "amount": 17})
        self.assertEqual(legalize("raise", 500, {"fold": True, "call": 10}), {"action": "call", "amount": 10})

    def test_table_state_reset(self):
        obs = observation()
        self.bot.act(obs)
        self.bot.stats[1][1] = 100
        obs["table_id"] = "another-table"
        self.bot.act(obs)
        self.assertEqual(self.bot.stats[1][1], 0)
        self.assertEqual(self.bot.stats[1][0], 1)

    def test_side_pot_excluded_from_pot_odds(self):
        obs = observation("7c 2d", "As Kh Qs 8d 4c", call=100, stack=100)
        obs["players"] = [
            {"stack": 100, "bet": 0, "committed": 100, "folded": False},
            {"stack": 0, "bet": 2000, "committed": 2100, "folded": False},
            {"stack": 0, "bet": 2000, "committed": 2100, "folded": False},
        ]
        obs["pot"] = 4300
        self.bot.act(obs)
        self.assertAlmostEqual(self.bot.last_info["pot_odds"], 100 / 600, places=3)

    def test_random_legal_states(self):
        rng = random.Random(774)
        for j in range(1500):
            nboard = rng.choice((0, 3, 4, 5))
            deal = rng.sample(range(52), nboard + 2)
            stack = rng.randrange(1, 10000)
            call = rng.randrange(stack + 1)
            obs = observation(call=call, stack=stack)
            obs.update(hole_cards=deal[:2], board=deal[2:], hand_id=j)
            obs["players"][2].update(stack=10000, bet=call, committed=call)
            action = self.bot.act(obs)
            self.assertIsNone(self.bot.last_error)
            legal = obs["legal_actions"]
            self.assertIn(action["action"], legal)
            if action["action"] == "raise":
                self.assertTrue(legal["raise"]["min"] <= action["amount"] <= legal["raise"]["max"])
            if action["action"] == "call":
                self.assertEqual(action["amount"], legal["call"])


class EngineTests(unittest.TestCase):
    def test_sidepots_and_uncalled_excess(self):
        board = cards("2c 3d 7s 9h Jc")
        players = [
            {"committed": 100, "folded": False, "hole": cards("As Ah")},
            {"committed": 300, "folded": False, "hole": cards("Ks Kh")},
            {"committed": 500, "folded": False, "hole": cards("Qs Qh")},
            {"committed": 300, "folded": True, "hole": cards("Ts Th")},
        ]
        self.assertEqual(award_pots(players, board), [400, 600, 200, 0])

    def test_tie_odd_chip_left_of_button(self):
        board = cards("As Ks Qs Js Ts")
        players = [{"committed": 5, "folded": False, "hole": cards("2c 3d")}, {"committed": 5, "folded": False, "hole": cards("4c 5d")}, {"committed": 5, "folded": True, "hole": cards("6c 7d")}]
        self.assertEqual(award_pots(players, board, button=0), [7, 8, 0])

    def test_cumulative_short_raise_reopens(self):
        t = Table([Baseline("fold", 1) for _ in range(3)])
        t.players[0].update(bet=100, committed=100)
        t.current = 150
        t.last_full_raise = 100
        t.acted_to = {0: 100}
        self.assertNotIn("raise", t.legal_actions(0))
        t.current = 200
        self.assertIn("raise", t.legal_actions(0))
        t.acted_to[0] = 150
        self.assertNotIn("raise", t.legal_actions(0))

    def test_short_big_blind_keeps_full_bring_in(self):
        seen = []
        t = Table([Baseline("fold", 1) for _ in range(4)], seed=0, capture=lambda o, a, u: seen.append(o))
        # play_hand moves button to seat 1, so seat 3 posts the big blind.
        t.players[3]["stack"] = 5
        t.total_chips -= 1995
        t.play_hand()
        self.assertEqual(seen[0]["legal_actions"]["call"], 20)

    def test_heads_up_blinds_and_order(self):
        seen = []
        t = Table([Baseline("station", 1), Baseline("station", 2)], hand_limit=1, capture=lambda o, a, u: seen.append(o))
        t.play_hand()
        first = next(o for o in seen if o["street"] == "preflop")
        self.assertEqual(first["hero"], first["button"])
        self.assertEqual(first["players"][first["hero"]]["bet"], 10)
        post = next(o for o in seen if o["street"] == "flop")
        self.assertNotEqual(post["hero"], post["button"])

    def test_complete_tables_conserve_chips(self):
        for seed in range(12):
            bots = [Baseline(("random", "tight", "pressure", "station")[i % 4], seed * 8 + i) for i in range(8)]
            table = Table(bots, seed=seed, hand_limit=200, payouts=(.5, .3, .2, 0, 0, 0, 0, 0))
            table.run()
            self.assertEqual(sum(p["stack"] for p in table.players), 16000)
            self.assertAlmostEqual(sum(table.points), 1)


if __name__ == "__main__":
    unittest.main()
