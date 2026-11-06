import itertools
import json
import random
import unittest

from pokerbot import Agent, Observation, Player
from pokerbot.cards import card, class_index, evaluate, evaluate_two, board_state
from pokerbot.equity import preflop_equity, postflop_equity
from pokerbot.icm import equities, focal_equity, bubble_factor
from pokerbot.fast_equity import features, equity as fast_equity
from pokerbot.model import Action, Event, legalize
from tests.reference import packed
from tools.simulator import Table
from tools.baselines import CallingStation, CheckFold, Shove, TightAggressive


def cards(s):
    return tuple(map(card, s.split()))


def observation(**overrides):
    data = dict(hole=cards("As Ad"), board=(), hero=0, button=5,
                players=tuple(Player(i, 2000 - (10 if i == 6 else 20 if i == 7 else 0),
                                     10 if i == 6 else 20 if i == 7 else 0,
                                     10 if i == 6 else 20 if i == 7 else 0) for i in range(8)),
                pot=30, big_blind=20, to_call=20, min_raise_to=40,
                max_raise_to=2000, hand_id=1, table_id="test")
    data.update(overrides)
    return Observation(**data)


class EvaluatorTests(unittest.TestCase):
    def test_feature_audit_and_fast_equity_boundaries(self):
        from pathlib import Path
        for line in Path("assets/equity_audit.jsonl").read_text().splitlines():
            d = json.loads(line)
            self.assertEqual(features(d["hole"], d["board"], d["range_index"], d["pressure"]), tuple(d["features"]))
        self.assertEqual(fast_equity(cards("As Ks"), cards("Qs Js Ts 2d 3d"), .1, 7, .9)[0], 1)
        self.assertEqual(fast_equity(cards("2c 3d"), cards("As Ks Qs Js Ts"), 1, 7, 0)[0], 1 / 8)

    def test_compiled_model_matches_numeric_tree_traversal(self):
        from pathlib import Path
        from pokerbot._equity_compiled import predict_raw
        model = json.loads(Path("assets/equity_model.json").read_text())
        for line in Path("assets/equity_audit.jsonl").read_text().splitlines()[::4]:
            d = json.loads(line)
            x = d["features"]
            raw = list(model["base"])
            for tree in model["trees"]:
                node = 0
                for _ in range(6):
                    node = 2 * node + 1 + (x[tree["features"][node]] > tree["thresholds"][node])
                for j in range(5):
                    raw[j] += model["rate"] * tree["leaves"][node - 63][j]
            expected = [raw[0], raw[1], raw[2], (raw[2]+raw[3])/2,
                        raw[3], (raw[3]+raw[4])/2, raw[4]]
            for j in range(7):
                actual_hu, actual_eq = predict_raw(x, j)
                self.assertAlmostEqual(actual_hu, raw[0], places=12)
                self.assertAlmostEqual(actual_eq, expected[j], places=12)

    def test_known_edge_cases(self):
        examples = ["As Ks Qs Js Ts 2d 3d", "As 2s 3s 4s 5s Kd Kc",
                    "As Ad Ac Ah Ks Kd Qh", "As Ad Ac Ks Kd Kc 2h",
                    "As Ad Ks Kd Qs Qd 2h", "As Ks 9s 6s 2s 3s 4s",
                    "As 2d 3h 4c 5d Qd Qs", "As Ad Ac 2d 3h 8s 9c"]
        for s in examples:
            c = cards(s)
            self.assertEqual(evaluate(c), packed(c), s)

    def test_reference_random_5_6_7_cards(self):
        rng = random.Random(89271)
        for n in (5, 6, 7):
            for _ in range(2500):
                c = tuple(rng.sample(range(52), n))
                expected = packed(c)
                self.assertEqual(evaluate(c), expected, c)
                self.assertEqual(evaluate_two(board_state(c[2:]), c[0], c[1]), expected, c)

    def test_card_and_class_conventions(self):
        self.assertEqual(card("2c"), 0)
        self.assertEqual(card("As"), 51)
        self.assertEqual(card("10h"), card("Th"))
        classes = {class_index(a, b) for a in range(52) for b in range(a + 1, 52)}
        self.assertEqual(len(classes), 169)
        with self.assertRaises(ValueError):
            card(52)

    def test_equity_sanity(self):
        self.assertTrue(.84 < preflop_equity(*cards("As Ah")) < .87)
        self.assertTrue(.32 < preflop_equity(*cards("7s 2h")) < .37)
        self.assertGreater(preflop_equity(*cards("As Ah"), opponents=7), .35)
        eq, hu, score = postflop_equity(cards("As Ks"), cards("Qs Js Ts 2d 3d"), .1, 7, .9, 7)
        self.assertEqual(eq, 1)
        self.assertEqual(hu, 1)
        eq, hu, _ = postflop_equity(cards("2c 3d"), cards("As Ks Qs Js Ts"), 1, 7, 0, 9)
        self.assertAlmostEqual(eq, 1 / 8)
        self.assertEqual(hu, .5)


class ICMTests(unittest.TestCase):
    def test_focal_matches_independent_subset_dp(self):
        rng = random.Random(71263)
        for n in range(2, 9):
            for _ in range(25):
                stacks = [rng.randrange(1, 16000) for i in range(n)]
                payouts = sorted((rng.random() for i in range(n)), reverse=True)
                expected = equities(stacks, payouts)
                for i in range(n):
                    self.assertAlmostEqual(focal_equity(stacks, i, payouts), expected[i], places=10)
                self.assertAlmostEqual(sum(expected), sum(payouts), places=10)

    def test_winner_take_all_and_bubble(self):
        stacks = [2000] * 8
        self.assertAlmostEqual(bubble_factor(stacks, 0, 1, [1] + [0] * 7, 2000), 1)
        self.assertGreater(bubble_factor([2000, 2000, 2000, 300], 0, 1, [50, 30, 20, 0], 2000), 1.5)
        self.assertEqual(focal_equity([0, 100, 200], 0, [50, 30, 20]), 20)

    def test_paired_icm_matches_three_independent_evaluations(self):
        rng = random.Random(17271)
        for n in range(2, 9):
            for _ in range(40):
                stacks = [rng.randrange(10, 8000) for i in range(n)]
                payouts = sorted((rng.random() for i in range(n)), reverse=True)
                wager = rng.choice((.2, .65, 1.0)) * min(stacks[0], stacks[1])
                base = focal_equity(stacks, 0, payouts)
                win, lose = stacks[:], stacks[:]
                win[0] += wager
                win[1] -= wager
                lose[0] -= wager
                lose[1] += wager
                gain = focal_equity(win, 0, payouts) - base
                loss = base - focal_equity(lose, 0, payouts)
                expected = max(1.0, min(100.0, loss / max(gain, 1e-12)))
                self.assertAlmostEqual(bubble_factor(stacks, 0, 1, payouts, wager), expected, places=9)


class PolicyTests(unittest.TestCase):
    def test_preflop_premiums_and_trash(self):
        agent = Agent()
        self.assertEqual(agent.act(observation()).kind, "raise")
        self.assertEqual(agent.act(observation(hole=cards("7c 2d"))).kind, "fold")

    def test_premiums_respect_extreme_flat_payout_bubble(self):
        ps = (Player(0, 6000), Player(1, 5000, 1000, 1000),
              Player(2, 1, folded=True), Player(3, 1, folded=True))
        obs = observation(players=ps, button=3, pot=1000, to_call=1000,
                          min_raise_to=2000, max_raise_to=6000,
                          history=(Event(1, 0, "raise", 1000),),
                          payouts=(1, 1, 1, 0))
        self.assertEqual(Agent().act(obs).kind, "fold")

    def test_nuts_do_not_fold_when_raise_closed(self):
        players = (Player(0, 900, 100, 100), Player(1, 0, 1000, 1000))
        obs = observation(hole=cards("As Ks"), board=cards("Qs Js Ts 3d 4d"),
                          players=players, hero=0, button=1, to_call=900,
                          pot=1100, min_raise_to=1900, max_raise_to=1000,
                          can_raise=False, legal=("fold", "call"))
        self.assertEqual(Agent().act(obs).kind, "call")

    def test_royal_board_chop_never_folds_due_to_icm(self):
        ps = (Player(0, 1900, 100, 100), Player(1, 0, 2000, 2000), Player(2, 100, 0, 0, folded=True))
        obs = observation(hole=cards("2c 3d"), board=cards("As Ks Qs Js Ts"),
                          players=ps, hero=0, button=1, to_call=1900, pot=2100,
                          min_raise_to=3900, max_raise_to=2000, can_raise=False,
                          payouts=(50, 30, 0), legal=("fold", "call"))
        self.assertEqual(Agent().act(obs).kind, "call")

    def test_legal_gate(self):
        obs = observation(to_call=0, legal=("check",))
        self.assertEqual(legalize(obs, "fold"), Action("check"))
        obs = observation(min_raise_to=2400)
        self.assertEqual(legalize(obs, "raise", 3000), Action("raise", 2000))
        obs = observation(can_raise=False, legal=("fold", "call"))
        self.assertEqual(legalize(obs, "raise", 200), Action("call"))

    def test_model_validation_and_json_roundtrip(self):
        from dataclasses import asdict
        obs = observation()
        d = json.loads(json.dumps(asdict(obs)))
        self.assertEqual(Observation.from_dict(d), obs)
        d["hole"] = ["As", "As"]
        with self.assertRaises(ValueError):
            Observation.from_dict(d)

    def test_idempotent_stats_and_table_reset(self):
        agent = Agent()
        obs = observation(history=(Event(1, 0, "raise", 50),))
        agent.act(obs)
        agent.act(obs)
        self.assertEqual(agent.stats[1], [1, 1, 1])
        agent.act(observation(table_id="other", history=()))
        self.assertEqual(agent.stats[1], [0, 0, 0])

    def test_random_legal_observations(self):
        rng = random.Random(80463)
        agent = Agent()
        for i in range(1200):
            n = rng.choice((0, 3, 4, 5))
            c = rng.sample(range(52), n + 2)
            stack = rng.randrange(1, 3000)
            own = rng.randrange(0, 200)
            opp = rng.randrange(own, 500)
            ps = (Player(0, stack, own, own + 50), Player(1, 3000, opp, opp + 50))
            call = min(stack, opp - own)
            can_raise = rng.choice((True, False))
            legal = ("fold", "call") if call else ("check",)
            if can_raise:
                legal += ("raise",)
            obs = observation(hole=tuple(c[:2]), board=tuple(c[2:]), players=ps,
                              hero=0, button=1, pot=own + opp + 100,
                              to_call=call, min_raise_to=opp + 20,
                              max_raise_to=own + stack, can_raise=can_raise,
                              legal=legal, hand_id=i)
            obs.validate()
            Table.validate_action(obs, agent.act(obs))


class EngineTests(unittest.TestCase):
    def test_short_blinds_with_one_live_stack_need_no_uncontested_call(self):
        class Record:
            def __init__(self):
                self.observations = []
            def act(self, obs):
                self.observations.append(obs)
                return legalize(obs, "call")
        hero = Record()
        t = Table([hero, CallingStation()], starting_stack=2000)
        t.stacks, t.total_chips, t.button = [2000, 5], 2005, 1
        t.play_hand()
        self.assertEqual(t.decisions, 0)
        self.assertEqual(sum(t.stacks), 2005)
        t = Table([hero, CallingStation(), CallingStation()], starting_stack=2000)
        t.stacks, t.total_chips, t.button = [2000, 5, 10], 2015, 2
        t.play_hand()
        self.assertEqual(hero.observations[-1].to_call, 10)
        self.assertEqual(sum(t.stacks), 2015)

    def test_deadline_fallback(self):
        r = Table([Shove()] * 8, hand_cap=4, decision_limit_us=0).play()
        self.assertGreater(sum(r.timeouts), 0)
        self.assertEqual(r.errors, 0)
        self.assertEqual(sum(r.stacks), 16000)

    def test_short_allin_does_not_reopen_but_cumulative_does(self):
        class Script:
            def __init__(self, first):
                self.first = first
                self.observations = []
            def act(self, obs):
                self.observations.append(obs)
                if len(self.observations) == 1 and self.first:
                    return legalize(obs, "raise", self.first)
                return legalize(obs, "call")
        def run(stacks, scripts):
            t = Table(scripts, starting_stack=1000)
            t.stacks = stacks[:]
            t.starting = stacks[:]
            t.total = [0] * len(stacks)
            t.bets = [0] * len(stacks)
            t.folded = [False] * len(stacks)
            t.bb = 100
            t.history = []
            t.board = cards("2c 4d 6h")
            t.holes = [cards(s) for s in ("As Ad", "Ks Kd", "Qs Qd", "Js Jd")[:len(stacks)]]
            t.round(1, 0, 0)
            return t
        scripts = [Script(100), Script(0), Script(150)]
        run([1000, 1000, 150], scripts)
        self.assertFalse(scripts[0].observations[1].can_raise)
        scripts = [Script(100), Script(0), Script(150), Script(200)]
        run([1000, 1000, 150, 200], scripts)
        self.assertTrue(scripts[0].observations[1].can_raise)
        self.assertEqual(scripts[0].observations[1].min_raise_to, 300)

    def test_short_opening_minimum_raise_and_prior_check(self):
        class Script:
            def __init__(self, amount=0):
                self.amount, self.observations = amount, []
            def act(self, obs):
                self.observations.append(obs)
                return legalize(obs, "raise", self.amount) if self.amount else legalize(obs, "call")
        scripts = [Script(), Script(50), Script()]
        t = Table(scripts, starting_stack=1000)
        t.stacks = [1000, 50, 1000]
        t.starting = t.stacks[:]
        t.total, t.bets, t.folded = [0]*3, [0]*3, [False]*3
        t.bb, t.history = 100, []
        t.board = cards("2c 4d 6h")
        t.holes = [cards("As Ad"), cards("Ks Kd"), cards("Qs Qd")]
        t.round(1, 0, 0)
        self.assertEqual(scripts[2].observations[0].min_raise_to, 150)
        self.assertFalse(scripts[0].observations[1].can_raise)

    def test_side_pots_and_refund(self):
        t = Table([CallingStation()] * 3)
        t.total = [100, 200, 300]
        t.stacks = [0, 0, 0]
        t.folded = [False] * 3
        t.holes = [cards("As Ad"), cards("Ks Kd"), cards("Qs Qd")]
        t.runout = cards("2c 4d 6h 8s Tc")
        t.settle()
        self.assertEqual(t.stacks, [300, 200, 100])

    def test_tournament_conservation_and_legal_actions(self):
        for seed in range(5):
            agents = [Agent(seed), TightAggressive(), CallingStation(), CheckFold(),
                      Shove(), TightAggressive(True), Agent(seed + 2), CallingStation()]
            t = Table(agents, seed=seed, hand_cap=80)
            result = t.play()
            self.assertEqual(sum(result.stacks), 16000)
            self.assertEqual(result.errors, 0)
            self.assertAlmostEqual(sum(result.points), 28)

    def test_all_station_and_all_fold(self):
        for agents in ([CallingStation()] * 8, [CheckFold()] * 8,
                       [Shove()] * 8, [Agent()] + [CheckFold()] * 7):
            r = Table(agents, seed=17, hand_cap=30).play()
            self.assertEqual(sum(r.stacks), 16000)
            self.assertEqual(r.errors, 0)


if __name__ == "__main__":
    unittest.main()
