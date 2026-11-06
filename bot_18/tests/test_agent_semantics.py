"""Independent contract/accounting regressions and valid-state policy fuzzing."""
from copy import deepcopy
import random
import statistics
from time import perf_counter_ns, process_time_ns
import unittest
from unittest.mock import patch

from agent import Agent
from equity.evaluator import evaluate
from equity.model import forced_tie
from policy.icm import call_threshold, value
from sim.engine import Hand, legal_action
from sim.icm import icm


def cards(text):
    return ["23456789TJQKA".index(c[0]) * 4 + "cdhs".index(c[1]) for c in text.split()]


def royal_river(n=2, stacks=None, payouts=None, board="As Ks Qs Js Ts"):
    h = Hand(stacks or [1000] * n, 0, payouts=payouts, rng=random.Random(781))
    while h.street != "river":
        o = h.observation()
        h.apply({"action": "call" if o["to_call"] else "check"})
    # Replace fully exposed cards only; all remaining cards are hidden and no
    # future board cards will be drawn. Unique physical holdings remain valid.
    h.board = cards(board)
    rest = [c for c in range(52) if c not in h.board]
    for i in range(n):
        h.holes[i] = rest[2 * i:2 * i + 2]
    return h


def sample_randomized_actions(hands=600, seed=181810):
    """Exercise real engine states; random play explores branches beyond policy."""
    rng = random.Random(seed)
    latencies, cpu, streets = [], [], set()
    short_raises = closed_raises = sidepots = 0
    total_fallbacks = 0
    for hand_id in range(hands):
        n = rng.randrange(2, 9)
        stacks = [rng.randrange(1, 5001) for _ in range(n)]
        bb = rng.choice([2, 20, 100, 500])
        pay = None if hand_id % 3 == 0 else ([50, 30, 20] + [0] * 8)[:n]
        hand = Hand(stacks, rng.randrange(n), bb // 2, bb, rng=rng,
                    hand_id=hand_id, payouts=pay, hands_remaining=rng.randrange(1, 201))
        agents = [Agent(seed=seed + hand_id * 8 + seat) for seat in range(n)]
        while not hand.done:
            obs = hand.observation()
            hero = obs["hero"]
            before = deepcopy(obs)
            started_wall, started_cpu = perf_counter_ns(), process_time_ns()
            action = agents[hero].act(obs)
            cpu.append(process_time_ns() - started_cpu)
            latencies.append(perf_counter_ns() - started_wall)
            if obs != before:
                raise AssertionError("policy mutated its observation")
            legal_action(obs, action)
            streets.add(obs["street"])
            short_raises += obs["can_raise"] and obs["min_raise_to"] > obs["max_raise_to"]
            closed_raises += not obs["can_raise"] and obs["to_call"] > 0
            passive = {"action": "call" if obs["to_call"] else "check"}
            choices = [passive] * 5 + [action, {"action": "fold"}]
            if obs["can_raise"]:
                choices.append({"action": "raise", "amount": obs["max_raise_to"]})
                if obs["min_raise_to"] <= obs["max_raise_to"]:
                    choices.extend([{"action": "raise", "amount": obs["min_raise_to"]},
                                    {"action": "raise", "amount": rng.randint(obs["min_raise_to"], obs["max_raise_to"])}])
            hand.apply(rng.choice(choices))
        total_fallbacks += sum(agent.errors for agent in agents)
        sidepots += len(hand.pots) > 1
    ordered = sorted(latencies)
    def pct(q):
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))] / 1000
    return {"seed": seed, "hands": hands, "decisions": len(latencies),
            "streets": sorted(streets), "short_allin_raise_opportunities": short_raises,
            "closed_raise_decisions": closed_raises, "sidepot_hands": sidepots,
            "internal_fallbacks": total_fallbacks, "wall_overruns_1ms": sum(t > 1_000_000 for t in latencies),
            "wall_us": {"p50": pct(.5), "p95": pct(.95), "p99": pct(.99), "p999": pct(.999), "max": max(latencies) / 1000},
            "cpu_us": {"mean": statistics.mean(cpu) / 1000, "max": max(cpu) / 1000},
            "timing_note": "Cold and warm policy calls; act only. CPU clock resolution is platform dependent. No timeout enforcement."}


class AgentSemanticsTests(unittest.TestCase):
    def test_guaranteed_chop_calls_allin_even_with_icm(self):
        boards = ("As Ks Qs Js Ts", "Ac Ad Ah As Ks", "7c 7d 7h 7s As", "Ac Kd Qh Js Tc")
        for board in boards:
            for payouts in (None, [100, 0], [50, 50]):
                with self.subTest(board=board, payouts=payouts):
                    hand = royal_river(payouts=payouts, board=board)
                    hand.apply({"action": "raise", "amount": 980})
                    obs = hand.observation()
                    self.assertTrue(forced_tie(obs["hole"], obs["board"]))
                    action = Agent().act(obs)
                    self.assertEqual(action["action"], "call")
                    hand.apply(action)
                    self.assertEqual(hand.stacks, [1000, 1000])

    def test_guaranteed_chop_multiway_unequal_sidepots(self):
        hand = royal_river(n=3, stacks=[1000, 400, 2000], payouts=[50, 30, 20])
        hand.apply({"action": "raise", "amount": 380})  # SB all-in
        hand.apply({"action": "raise", "amount": 1980}) # BB all-in
        obs = hand.observation()
        self.assertEqual(obs["to_call"], 980)
        action = Agent().act(obs)
        self.assertEqual(action["action"], "call")
        hand.apply(action)
        self.assertEqual(hand.stacks, [1000, 400, 2000])

    def test_guaranteed_chop_does_not_value_raise(self):
        hand = royal_river()
        obs = hand.observation()
        self.assertTrue(obs["can_raise"])
        self.assertEqual(Agent().act(obs)["action"], "check")

    def test_raise_sanitization_full_and_short_allin(self):
        for stacks in ([1000, 1000], [25, 1000]):
            hand = Hand(stacks, 0)
            obs = hand.observation()
            for target in (-10000, 0, 25, 40.75, 100000):
                action = Agent()._raise(obs, target)
                legal_action(obs, action)
                self.assertIsInstance(action["amount"], int)

    def test_raise_not_reopened_never_emitted(self):
        hand = Hand([1000, 150, 1000, 1000], 0)
        for action in ({"action": "raise", "amount": 100}, {"action": "call"},
                       {"action": "raise", "amount": 150}, {"action": "call"}):
            hand.apply(action)
        obs = hand.observation()
        self.assertFalse(obs["can_raise"])
        for hole in (cards("Ac Ad"), cards("7c 2d")):
            obs["hole"] = hole
            action = Agent().act(obs)
            self.assertNotEqual(action["action"], "raise")
            legal_action(obs, action)

    def test_icm_heads_up_chip_ev_matches_eligible_pot_odds(self):
        # A completely folded player's investment still belongs to the pot.
        players = [{"seat": 0, "stack": 900, "invested": 100},
                   {"seat": 1, "stack": 700, "invested": 300},
                   {"seat": 2, "stack": 950, "invested": 50}]
        self.assertAlmostEqual(call_threshold(players, 0, 1, 450, 200, [1, 0, 0]), 200 / 650)

    def test_icm_branch_accounting_against_independent_vector_model(self):
        rng = random.Random(1822)
        for n in range(2, 9):
            for _ in range(10):
                stacks = [rng.randrange(1, 3000) for _ in range(n)]
                call = rng.randrange(1, stacks[0] + 1)
                contributions = [rng.randrange(1, 100) for _ in range(n)]
                pot = sum(contributions)
                players = [{"seat": i, "stack": stack, "invested": contributions[i]} for i, stack in enumerate(stacks)]
                payouts = sorted([rng.randrange(101) for _ in range(n)], reverse=True)
                folded = stacks[:]
                folded[1] += pot
                won = stacks[:]
                won[0] += pot
                lost = stacks[:]
                lost[0] -= call
                lost[1] += pot + call
                self.assertEqual(sum(folded), sum(won))
                self.assertEqual(sum(won), sum(lost))
                f = icm(folded, payouts)[0]
                w = icm(won, payouts)[0]
                # Exact zero-stack hero means elimination, with last payout.
                l = payouts[-1] if lost[0] == 0 else icm(lost, payouts)[0]
                expected = 1.0 if w <= l + 1e-15 else min(1.0, max(0.0, (f - l) / (w - l)))
                self.assertAlmostEqual(call_threshold(players, 0, 1, pot, call, payouts), expected, places=10)

    def test_valid_randomized_observation_action_legality(self):
        report = sample_randomized_actions()
        self.assertEqual(report["internal_fallbacks"], 0)
        self.assertGreater(report["decisions"], 5000)
        self.assertEqual(set(report["streets"]), {"preflop", "flop", "turn", "river"})
        self.assertGreater(report["short_allin_raise_opportunities"], 0)
        self.assertGreater(report["closed_raise_decisions"], 0)
        self.assertGreater(report["sidepot_hands"], 100)

    def test_challenge_opponents_legal_without_candidate(self):
        # Only challenge agents face one another here. Candidate-vs-challenge
        # performance is deliberately left untouched for held-out evaluation.
        from sim.baselines import Baseline
        from sim.engine import tournament
        styles = ("range", "trap", "pressure")
        for seed in range(3):
            result = tournament([Baseline(styles[(seat + seed) % 3], seed=seat + 100 * seed) for seat in range(8)],
                                seed=98300 + seed, max_hands=50)
            self.assertEqual(sum(result.errors), 0)
            self.assertEqual(sum(result.illegal), 0)

    def test_benchmark_reports_internal_fallbacks_and_hashes(self):
        from sim.benchmark import run
        class InternalFallback:
            def __init__(self):
                self.errors = 0
            def reset(self):
                self.errors = 0
            def act(self, obs):
                self.errors += 1
                return {"action": "call" if obs["to_call"] else "check"}
        with patch("sim.benchmark.make_candidate", side_effect=lambda *_: InternalFallback()):
            report = run(tables=1, seed=98231, max_hands=3)
        self.assertEqual(report["candidate_fallbacks"], report["decisions"])
        self.assertGreater(report["candidate_fallbacks"], 0)
        self.assertEqual(report["errors"], 0)
        self.assertEqual(report["illegal"], 0)
        self.assertIn("agent.py", report["source_sha256"])
        self.assertEqual(report["source_changed_during_run"], [])

    def test_selfplay_uses_explicit_own_bot_configs(self):
        from sim.benchmark import run, SELFPLAY_CONFIGS
        report = run(candidate="baseline:tag", tables=1, seed=98232,
                     suite="selfplay", max_hands=1)
        self.assertEqual(report["selfplay_opponent_configs"], SELFPLAY_CONFIGS)
        self.assertEqual(set(report["rows"][0]["opponent_labels"]), set(SELFPLAY_CONFIGS))
        self.assertEqual(report["rows"][0]["opponent_errors"], 0)
        self.assertEqual(report["rows"][0]["opponent_illegal"], 0)
        self.assertEqual(report["opponent_fallbacks"], 0)


if __name__ == "__main__":
    unittest.main()
