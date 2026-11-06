"""Rule-focused tests for our independent local simulator."""
import random
import unittest

from lab.baselines import Aggressive, CallingStation, CheckFold, MinraisePressure, PremiumNit, RandomLegal, ShoveWide, TightValue
from lab.simulator import Hand, Tournament, advance_blinds, icm, reference_evaluate


class CallCheck:
    def __init__(self):
        self.observations = []

    def act(self, obs):
        self.observations.append(obs)
        return {"action": "call" if obs["to_call"] else "check"}


class Script(CallCheck):
    def __init__(self, script):
        super().__init__()
        self.script = list(script)

    def act(self, obs):
        self.observations.append(obs)
        if self.script:
            decision = self.script.pop(0)
            return decision(obs) if callable(decision) else decision
        return {"action": "call" if obs["to_call"] else "check"}


def raise_to(amount):
    return {"action": "raise", "amount": amount}


class SimulatorTests(unittest.TestCase):
    def test_heads_up_blinds_and_order(self):
        agents = [CallCheck(), CallCheck()]
        result = Hand([2000, 2000], agents, 0, 10, 20, rng=random.Random(1)).run()
        self.assertEqual([(e["street"], e["seat"]) for e in result.history[:2]], [("preflop", 0), ("preflop", 1)])
        self.assertEqual([e["seat"] for e in result.history if e["street"] == "flop"], [1, 0])
        initial = agents[0].observations[0]
        self.assertEqual((initial["players"][0]["bet"], initial["players"][1]["bet"], initial["to_call"], initial["pot"]), (10, 20, 10, 30))
        self.assertEqual(initial["min_raise_to"], 40)
        self.assertEqual(sum(result.stacks), 4000)

    def test_full_ring_order_and_no_hidden_cards(self):
        agents = [CallCheck() for _ in range(8)]
        result = Hand([2000] * 8, agents, 0, 10, 20, rng=random.Random(2)).run()
        self.assertEqual([e["seat"] for e in result.history if e["street"] == "preflop"], [3, 4, 5, 6, 7, 0, 1, 2])
        self.assertEqual([e["seat"] for e in result.history if e["street"] == "flop"], [1, 2, 3, 4, 5, 6, 7, 0])
        obs = agents[3].observations[0]
        self.assertEqual(len(obs["hole_cards"]), 2)
        self.assertEqual(obs["board"], [])
        self.assertTrue(all("hole" not in p and "hole_cards" not in p for p in obs["players"]))
        allcards = result.board + [c for hole in result.holes for c in hole]
        self.assertEqual(len(allcards), len(set(allcards)))

    def test_three_to_heads_up_never_repeats_big_blind(self):
        for busted in range(3):
            stacks = [100, 100, 100]
            stacks[busted] = 0
            button, sb, bb, anchor = advance_blinds(stacks, 1, 2)
            self.assertEqual(sb, button)
            self.assertEqual(anchor, button)
            self.assertEqual(stacks[button], 100)
            if stacks[2]:
                self.assertNotEqual(bb, 2)
            h = Hand(stacks, [CallCheck()] * 3, button, 10, 20, blind_seats=(sb, bb))
            self.assertEqual(h.bb_seat, bb)
            self.assertEqual(h.sb_seat, button)

    def test_dead_button_and_dead_small_blind(self):
        # Prior button0/SB1/BB2. Busting SB leaves a dead next button.
        button, sb, bb, anchor = advance_blinds([100, 0, 100, 100], 1, 2)
        self.assertEqual((button, sb, bb, anchor), (1, 2, 3, 2))
        h = Hand([100, 0, 100, 100], [CallCheck()] * 4, button, 10, 20, blind_seats=(sb, bb))
        self.assertEqual(h.observation(0)["button"], 1)
        self.assertEqual(h.observation(0)["small_blind_seat"], 2)
        self.assertEqual(h.observation(0)["big_blind_seat"], 3)
        self.assertEqual(sum(p.bet for p in h.players), 30)
        # Busting BB leaves a dead next SB; BB still progresses clockwise.
        button, sb, bb, anchor = advance_blinds([100, 100, 0, 100], 1, 2)
        self.assertEqual((button, sb, bb, anchor), (1, None, 3, 2))
        agents = [CallCheck() for _ in range(4)]
        result = Hand([100, 100, 0, 100], agents, button, 10, 20, blind_seats=(sb, bb)).run()
        self.assertEqual(agents[0].observations[0]["pot"], 20)
        self.assertEqual([e["seat"] for e in result.history if e["street"] == "preflop"], [0, 1, 3])
        self.assertEqual([e["seat"] for e in result.history if e["street"] == "flop"], [3, 0, 1])
        self.assertEqual(advance_blinds([100, 100, 0, 100], anchor, bb), (2, 3, 0, 3))

    def test_short_all_in_does_not_reopen(self):
        agents = [Script([raise_to(100)]), Script([raise_to(150)]), CallCheck()]
        result = Hand([2000, 150, 2000], agents, 0, 10, 20, rng=random.Random(3)).run()
        second = agents[0].observations[1]
        self.assertEqual(second["street"], "preflop")
        self.assertEqual(second["current_bet"], 150)
        self.assertEqual(second["min_raise_to"], 230)
        self.assertFalse(second["can_raise"])
        self.assertEqual(second["to_call"], 50)
        self.assertFalse(result.history[1]["full_raise"])
        self.assertEqual(sum(result.stacks), 4150)

    def test_cumulative_short_allins_reopen_per_player(self):
        agents = [CallCheck() for _ in range(5)]
        h = Hand([2000, 125, 2000, 200, 2000], agents, 4, 50, 100)
        # Begin an artificial postflop round while conserving every chip.
        for p in h.players:
            p.stack += p.committed
            p.bet = p.committed = 0
            p.all_in = False
        h.street = "flop"
        h.current_bet = 0
        h.last_full_raise = 100
        h.last_action_level = [None] * 5
        h.pending = set(range(5))
        for seat, action in [(0, raise_to(100)), (1, raise_to(125)), (2, {"action": "call"}), (3, raise_to(200)), (4, {"action": "call"})]:
            self.assertTrue(h._validate(seat, action)[0])
            h._apply(seat, action)
        self.assertTrue(h.observation(0)["can_raise"])
        self.assertFalse(h.observation(2)["can_raise"])
        self.assertEqual(h.observation(0)["min_raise_to"], 300)

    def test_short_opening_allin_keeps_full_min_raise(self):
        h = Hand([2000, 20, 2000], [CallCheck()] * 3, 0, 50, 100)
        for p in h.players:
            p.stack += p.committed
            p.bet = p.committed = 0
            p.all_in = False
        h.street = "flop"
        h.current_bet = 0
        h.last_full_raise = 100
        h.last_action_level = [None] * 3
        h.pending = {0, 1, 2}
        h._apply(0, {"action": "check"})
        h._apply(1, raise_to(20))
        self.assertEqual(h.observation(2)["min_raise_to"], 120)
        self.assertTrue(h.observation(2)["can_raise"])
        self.assertFalse(h.observation(0)["can_raise"])
        self.assertTrue(h._validate(2, raise_to(120))[0])
        self.assertFalse(h._validate(2, raise_to(100))[0])

    def test_raise_to_and_full_raise_increment(self):
        h = Hand([2000] * 3, [CallCheck()] * 3, 0, 10, 20)
        h.pending = {0, 1, 2}
        h._apply(0, raise_to(70))
        obs = h.observation(1)
        self.assertEqual(obs["min_raise_to"], 120)
        self.assertEqual(obs["to_call"], 60)
        self.assertEqual(obs["pot"], 100)
        self.assertEqual(h.history[-1]["amount"], 70)
        self.assertEqual(h.history[-1]["amount_paid"], 70)
        self.assertEqual(h.history[-1]["stack_before"], 2000)

    def test_big_blind_short_still_nominal_preflop_call(self):
        agents = [CallCheck(), CallCheck(), CallCheck()]
        h = Hand([100, 100, 7], agents, 0, 10, 20)
        obs = h.observation(0)
        self.assertEqual(obs["current_bet"], 20)
        self.assertEqual(obs["to_call"], 20)
        self.assertEqual(obs["min_raise_to"], 40)
        result = h.run()
        self.assertEqual(sum(result.stacks), 207)
        self.assertEqual(sum(p["amount"] for p in result.pots), 47)

    def test_heads_up_short_blind_needs_no_excess_call(self):
        agents = [CallCheck(), CallCheck()]
        result = Hand([100, 7], agents, 0, 10, 20).run()
        self.assertEqual(len(result.history), 0)
        self.assertEqual(len(result.board), 5)
        self.assertEqual(sum(p["amount"] for p in result.pots), 14)
        self.assertEqual(result.refunds, [{"seat": 0, "street": "preflop", "amount": 3}])
        self.assertEqual(sum(result.stacks), 107)

    def test_lone_actionable_only_calls_actual_short_blinds(self):
        agents = [CallCheck(), CallCheck(), CallCheck()]
        result = Hand([100, 5, 7], agents, 0, 10, 20).run()
        self.assertEqual(agents[0].observations[0]["to_call"], 7)
        self.assertEqual(sum(result.stacks), 112)
        self.assertEqual(sum(p["amount"] for p in result.pots), 19)

    def test_uncalled_raise_refund(self):
        agents = [Script([raise_to(2000)]), CheckFold(), CheckFold()]
        result = Hand([2000] * 3, agents, 0, 10, 20).run()
        self.assertEqual(result.stacks, [2030, 1990, 1980])
        self.assertEqual(result.refunds, [{"seat": 0, "street": "preflop", "amount": 1980}])
        self.assertEqual(sum(p["amount"] for p in result.pots), 50)

    def test_sidepots_and_uncalled_allin(self):
        h = Hand([100, 200, 300], [CallCheck()] * 3, 0, 10, 20)
        for p, amount, hole in zip(h.players, [100, 200, 300], [[48, 49], [44, 45], [40, 41]]):
            p.stack = 0
            p.bet = p.committed = amount
            p.hole = hole
            p.all_in = True
        h.board = [0, 5, 22, 31, 36]
        h._refund_uncalled()
        h._settle_showdown()
        self.assertEqual([p.stack for p in h.players], [300, 200, 100])
        self.assertEqual([pot["amount"] for pot in h.pots], [300, 200])
        self.assertEqual(h.refunds[0]["amount"], 100)

    def test_split_pot_odd_chip_left_of_button(self):
        h = Hand([10, 10, 10], [CallCheck()] * 3, 0, 1, 2)
        for p, amount, hole in zip(h.players, [2, 2, 1], [[0, 4], [1, 5], [2, 6]]):
            p.stack = 10 - amount
            p.bet = p.committed = amount
            p.hole = hole
        h.players[2].folded = True
        h.board = [34, 38, 42, 46, 50]  # Royal flush on board.
        h._settle_showdown()
        self.assertEqual([p.stack for p in h.players], [10, 11, 9])

    def test_allin_runout_no_meaningless_betting(self):
        agents = [Script([raise_to(2000)]), CallCheck()]
        result = Hand([2000, 100], agents, 0, 10, 20).run()
        self.assertEqual(len(result.board), 5)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(sum(result.stacks), 2100)
        self.assertEqual(result.refunds[0]["amount"], 1900)

    def test_illegal_actions_and_exception_fall_back(self):
        class Broken:
            def act(self, obs):
                raise RuntimeError("intentional test failure")
        agents = [Script([raise_to(21)]), Broken(), CallCheck()]
        result = Hand([2000] * 3, agents, 0, 10, 20).run()
        self.assertEqual(result.history[0]["action"], "fold")
        self.assertEqual(result.decisions[0].fallback, "invalid")
        self.assertEqual(result.decisions[1].fallback, "exception")
        self.assertEqual(sum(result.stacks), 6000)

    def test_deadline_falls_back(self):
        agents = [CallCheck(), CallCheck()]
        result = Hand([2000, 2000], agents, 0, 10, 20, enforce_deadline=True, deadline_ns=0).run()
        self.assertEqual(result.history[0]["action"], "fold")
        self.assertTrue(all(r.fallback == "deadline" for r in result.decisions))

    def test_reference_hands(self):
        self.assertEqual(reference_evaluate([48, 0, 4, 8, 12])[0:2], (8, 5))
        self.assertGreater(reference_evaluate([48, 49, 50, 51, 0]), reference_evaluate([44, 45, 46, 48, 49]))
        self.assertEqual(reference_evaluate([0, 4, 8, 12, 16, 49, 50]), reference_evaluate([0, 4, 8, 12, 16, 41, 42]))

    def test_icm_symmetry_and_chip_linear_payout(self):
        result = icm([2000] * 8, [50, 30, 20, 0, 0, 0, 0, 0])
        for value in result:
            self.assertAlmostEqual(value, 12.5)
        self.assertEqual(icm([1, 3], [1, 0]), [0.25, 0.75])
        self.assertAlmostEqual(sum(icm([3, 2, 1], [9, 5, 2])), 16)

    def test_tournament_chip_and_point_conservation(self):
        agents = [RandomLegal(i) for i in range(8)]
        result = Tournament(agents, blind_schedule=[(10, 20), (50, 100), (100, 200), (200, 400)], stacks=2000, payouts=[50, 30, 20, 0, 0, 0, 0, 0], seed=41).run()
        self.assertEqual(sum(result.stacks), 16000)
        self.assertAlmostEqual(sum(result.points), 100)
        self.assertLessEqual(result.hands, 200)
        self.assertTrue(all(record.fallback is None for record in result.decisions))

    def test_fuzz_chip_conservation_and_baseline_legality(self):
        for seed in range(80):
            rng = random.Random(seed)
            n = 2 + seed % 7
            styles = [RandomLegal, CallingStation, TightValue, Aggressive]
            agents = [styles[(seed + i) % len(styles)](seed * 11 + i) for i in range(n)]
            stacks = [rng.randrange(1, 3000) for _ in range(n)]
            result = Hand(stacks, agents, seed % n, 10, 20, rng=rng).run()
            self.assertEqual(sum(result.stacks), sum(stacks))
            self.assertTrue(all(p >= 0 for p in result.stacks))
            self.assertTrue(all(record.fallback is None for record in result.decisions), (seed, result.decisions))

    def test_stress_preflop_profiles(self):
        from itertools import combinations
        from lab.baselines import _hand_key
        wide = ShoveWide()
        width = sum(wide.percentiles[_hand_key(hole)] <= 0.60 for hole in combinations(range(52), 2)) / 1326
        self.assertAlmostEqual(width, 0.60, delta=0.01)
        h = Hand([2000] * 3, [CallCheck()] * 3, 0, 10, 20)
        obs = h.observation(0)
        obs["hole_cards"] = [48, 49]
        self.assertEqual(wide.act(obs), raise_to(2000))
        self.assertEqual(PremiumNit().act(obs), raise_to(60))
        obs["hole_cards"] = [20, 1]  # 72 offsuit.
        self.assertEqual(wide.act(obs), {"action": "fold"})
        self.assertEqual(PremiumNit().act(obs), {"action": "fold"})
        obs["hole_cards"] = [48, 49]
        obs["current_bet"] = 1000
        obs["to_call"] = 1000
        obs["min_raise_to"] = 1980
        self.assertEqual(PremiumNit().act(obs), {"action": "call"})
        obs["hole_cards"] = [40, 41]  # QQ rejects a deep huge bet for this nit.
        self.assertEqual(PremiumNit().act(obs), {"action": "fold"})

    def test_minraise_pressure_uses_legal_minimum_open(self):
        h = Hand([2000] * 3, [CallCheck()] * 3, 0, 10, 20)
        obs = h.observation(0)
        obs["hole_cards"] = [48, 49]
        self.assertEqual(MinraisePressure(1).act(obs), raise_to(40))

    def test_stress_baselines_fuzz_legality_and_conservation(self):
        styles = [ShoveWide, PremiumNit, MinraisePressure]
        for seed in range(120):
            rng = random.Random(50000 + seed)
            n = 2 + seed % 7
            agents = [styles[(seed + i) % len(styles)](seed * 11 + i) for i in range(n)]
            stacks = [rng.randrange(1, 3000) for _ in range(n)]
            result = Hand(stacks, agents, seed % n, 10, 20, rng=rng).run()
            self.assertEqual(sum(result.stacks), sum(stacks))
            self.assertTrue(all(record.fallback is None for record in result.decisions), (seed, result.decisions))


if __name__ == "__main__":
    unittest.main()
