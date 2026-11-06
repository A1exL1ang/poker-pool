import json
from pathlib import Path
import random
import unittest
from bot6.agent import Agent
from bot6.cards import parse_card, evaluate_mask, CARD_MASK
from bot6.equity import estimate, preflop_equity, RANGES, hand_quality, texture, likelihood
from bot6.interface import State, legalize, eligible_pots
from bot6.icm import equity, risk_premium, allin_call_threshold


def observation(hole=("As", "Ah"), board=(), call=20, pot=30, stack=2000, opponent=1980, bet=20):
    return {"hero": 0, "hole_cards": list(hole), "board": list(board), "button": 0,
            "big_blind": 20, "pot": pot, "to_call": call, "min_raise_to": bet + 20,
            "max_raise_to": stack, "table_id": 1, "hand_id": 1,
            "legal_actions": ["fold", "call", "raise"] if call else ["check", "raise"],
            "players": [{"seat": 0, "stack": stack, "bet": 0, "contributed": 0},
                        {"seat": 1, "stack": opponent, "bet": bet, "contributed": bet, "all_in": opponent == 0}],
            "history": []}


class InterfaceTest(unittest.TestCase):
    def test_unwinnable_excess_excluded_from_pot_odds(self):
        o = observation(call=100, stack=100, bet=1000, pot=1000, opponent=0)
        total, layers = eligible_pots(State.from_dict(o))
        self.assertEqual(total, 200)
        self.assertIsNone(layers)

    def test_sidepot_eligibility(self):
        o = observation(call=0, stack=100, bet=0, pot=700)
        o["players"] = [
            {"seat": 0, "stack": 100, "contributed": 300},
            {"seat": 1, "stack": 100, "contributed": 300},
            {"seat": 2, "stack": 0, "contributed": 100, "all_in": True}]
        total, layers = eligible_pots(State.from_dict(o))
        self.assertEqual(total, 700)
        self.assertEqual(layers, [(300, 3), (400, 1)])

    def test_short_allin_raise(self):
        o = observation(stack=35)
        s = State.from_dict(o)
        self.assertEqual(legalize(s, "raise", 500), {"action": "raise", "amount": 35})

    def test_raise_not_reopened(self):
        o = observation()
        o["can_raise"] = False
        self.assertNotEqual(Agent().act(o)["action"], "raise")

    def test_call_capped_at_stack(self):
        o = observation(call=100, stack=40, bet=100)
        self.assertEqual(State.from_dict(o).to_call, 40)

    def test_duplicate_and_bad_payouts(self):
        o = observation(hole=("As", "As"))
        with self.assertRaises(ValueError):
            State.from_dict(o)
        o = observation()
        o["payouts"] = [1, 2]
        with self.assertRaises(ValueError):
            State.from_dict(o)

    def test_captured_state_legality(self):
        path = Path("benchmarks/states.jsonl")
        if not path.exists():
            self.skipTest("Run simulator --capture for additional fixture coverage")
        agent = Agent()
        for line in path.read_text().splitlines()[:5000]:
            o = json.loads(line)
            a = agent.act(o)
            self.assertIn(a["action"], o["legal_actions"])
            if a["action"] == "raise":
                self.assertGreater(a["amount"], max(p["bet"] for p in o["players"]))
                self.assertLessEqual(a["amount"], o["max_raise_to"])
                self.assertTrue(a["amount"] >= o["min_raise_to"] or a["amount"] == o["max_raise_to"])


class StrategyTest(unittest.TestCase):
    def test_unobserved_players_are_not_counted_as_folds(self):
        a = Agent()
        o = observation()
        a.act(o)
        o["hand_id"] = 2
        a.act(o)
        self.assertNotIn(1, a.stats)
        o["history"] = [{"seat": 1, "street": "preflop", "action": "raise"}]
        a.act(o)
        o["hand_id"] = 3
        o["history"] = []
        a.act(o)
        self.assertEqual(a.stats[1], (1, 1, 1))

    def test_aces_raise(self):
        self.assertEqual(Agent().act(observation())["action"], "raise")

    def test_early_opening_avoids_dominated_offsuit_ace(self):
        o = observation()
        o.update({"button": 5, "to_call": 20, "pot": 30})
        o["players"] = [{"seat": i, "stack": 1980 if i == 7 else 1990 if i == 6 else 2000,
                         "bet": 20 if i == 7 else 10 if i == 6 else 0} for i in range(8)]
        o["hole_cards"] = ["As", "9h"]
        self.assertEqual(Agent().act(o)["action"], "fold")
        o["hole_cards"] = ["Js", "Ts"]
        self.assertEqual(Agent().act(o)["action"], "raise")

    def test_trash_folds_deep_jam(self):
        o = observation(hole=("7s", "2h"), call=2000, pot=2020, opponent=0, bet=2000)
        o["history"] = [{"seat": 1, "street": "preflop", "action": "raise", "amount": 2000}]
        self.assertEqual(Agent().act(o)["action"], "fold")

    def test_aces_call_jam(self):
        o = observation(call=2000, pot=2020, opponent=0, bet=2000)
        o["history"] = [{"seat": 1, "street": "preflop", "action": "raise", "amount": 2000}]
        self.assertEqual(Agent().act(o)["action"], "call")

    def test_nuts_never_fold(self):
        for seed in range(30):
            o = observation(hole=("As", "Ks"), board=("Qs", "Js", "Ts", "2d", "3c"), call=200, pot=500, bet=200)
            self.assertNotEqual(Agent(seed).act(o)["action"], "fold")

    def test_allin_opponents_no_bet(self):
        o = observation(board=("Ac", "Kd", "7s"), call=0, pot=200, opponent=0, bet=0)
        self.assertEqual(Agent().act(o)["action"], "check")

    def test_table_reset(self):
        a = Agent()
        a.act(observation())
        a.stats[1] = (100, 100, 100)
        o = observation()
        o["table_id"] = 2
        a.act(o)
        self.assertEqual(a.stats, {})

    def test_seed_reproducibility(self):
        a, b = Agent(60, time_budget_us=0), Agent(60, time_budget_us=0)
        o = observation(hole=("As", "Kh"), board=("Ad", "Ts", "6c"), call=0, pot=200, bet=0)
        for _ in range(20):
            self.assertEqual(a.act(o), b.act(o))


class EquityTest(unittest.TestCase):
    def test_ranges_include_complete_suit_classes(self):
        from collections import Counter
        for combos in RANGES:
            counts = Counter(c[3] for c in combos)
            for name, count in counts.items():
                self.assertEqual(count, 6 if len(name) == 2 else 4 if name[-1] == "s" else 12)

    def test_preflop_sanity(self):
        aa = tuple(map(parse_card, ("As", "Ah")))
        trash = tuple(map(parse_card, ("7s", "2h")))
        self.assertTrue(.84 < preflop_equity(aa, 1.) < .87)
        self.assertTrue(.32 < preflop_equity(trash, 1.) < .37)

    def test_board_royal_is_split(self):
        board = tuple(map(parse_card, "As Ks Qs Js Ts".split()))
        hole = tuple(map(parse_card, ("2c", "3d")))
        for n in (1, 2, 7):
            result, _ = estimate(hole, board, [(1., 0)] * n, random.Random(7))
            self.assertAlmostEqual(result, 1 / (n + 1))

    def test_split_main_and_sidepot(self):
        board = tuple(map(parse_card, "As Ks Qs Js Ts".split()))
        hole = tuple(map(parse_card, ("2c", "3d")))
        result, _ = estimate(hole, board, [(1., 0)] * 2, random.Random(7), pots=[(300, 3), (400, 1)])
        self.assertAlmostEqual(result, 300 / 700)

    def test_weighted_river_against_enumeration(self):
        board = tuple(map(parse_card, "Ad Ts 6c 3h 2d".split()))
        hole = tuple(map(parse_card, ("As", "Kh")))
        bm, hm = sum(CARD_MASK[c] for c in board), sum(CARD_MASK[c] for c in hole)
        hs = evaluate_mask(bm | hm)
        rm, bh, _ = texture(board)
        for pressure in (0, 1, 2):
            total = wins = 0.
            for a, b, cm, _ in RANGES[3]:
                if cm & (bm | hm):
                    continue
                made, draw = hand_quality(a, b, bm, rm, bh, len(board))
                weight = likelihood(made, draw, pressure)
                vs = evaluate_mask(bm | cm)
                total += weight
                wins += weight * (1 if hs > vs else .5 if hs == vs else 0)
            exact = wins / total
            rng = random.Random(606 + pressure)
            estimates = [estimate(hole, board, [(.4, pressure)], rng)[0] for _ in range(500)]
            self.assertAlmostEqual(sum(estimates) / len(estimates), exact, delta=.035)


class ICMTest(unittest.TestCase):
    def test_allin_wta_threshold_matches_odds(self):
        o = observation(call=100, stack=100, opponent=0, bet=100, pot=100)
        o["payouts"] = [1, 0]
        s = State.from_dict(o)
        self.assertAlmostEqual(allin_call_threshold(s, 200), .5)

    def test_bubble_threshold_tighter(self):
        o = observation(call=100, stack=100, opponent=0, bet=100, pot=100)
        o["players"].append({"seat": 2, "stack": 100, "folded": True})
        o["payouts"] = [1, 1, 0]
        s = State.from_dict(o)
        self.assertGreater(allin_call_threshold(s, 200), .5)

    def test_winner_take_all(self):
        stacks = [100, 200, 300, 400]
        for i, stack in enumerate(stacks):
            self.assertAlmostEqual(equity(stacks, [1, 0, 0, 0], i), stack / sum(stacks))

    def test_symmetry_conservation(self):
        stacks, payouts = [400, 300, 200, 100], [50, 30, 20, 0]
        values = [equity(stacks, payouts, i) for i in range(4)]
        self.assertAlmostEqual(sum(values), sum(payouts))
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        for i in range(4):
            self.assertAlmostEqual(equity([100] * 4, payouts, i), 25)

    def test_zero_stack_and_flat_payout(self):
        self.assertEqual(equity([10, 0, 0], [50, 30, 20], 1), 25)
        self.assertAlmostEqual(equity([1, 2, 3], [5, 5, 5], 2), 5)


if __name__ == "__main__":
    unittest.main()
