import itertools
import hashlib
import json
from pathlib import Path
import random
import subprocess
import unittest
from dataclasses import replace
from collections import Counter

from agent import Agent, State, Player, Event, Action, legalize, validate_state
from bot.cards import card, evaluate, features, preflop_index
from bot.model import PREFLOP, predict, predict_reference
from bot.icm import icm, risk_premium
from tools.simulate import Table, Baseline, distribute

ROOT = Path(__file__).resolve().parent.parent


def reference5(cards):
    """Slow independent five-card evaluator, returning category/kicker tuple."""
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    counts = Counter(ranks)
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    straight = 0
    if len(counts) == 5:
        if ranks[0] - ranks[4] == 4:
            straight = ranks[0]
        elif ranks == [14, 5, 4, 3, 2]:
            straight = 5
    if flush and straight:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if [g[0] for g in groups] == [3, 2]:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *ranks)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    if [g[0] for g in groups[:2]] == [2, 2]:
        return (2, max(groups[0][1], groups[1][1]), min(groups[0][1], groups[1][1]), groups[2][1])
    if groups[0][0] == 2:
        return (1, groups[0][1], *[r for r in ranks if r != groups[0][1]])
    return (0, *ranks)


def pack_reference(t):
    return (t[0] << 20) + sum(r << (16 - 4*i) for i, r in enumerate(t[1:]))


def make_state(**kw):
    d = dict(table_id="test", hand_id=0, hero=0, button=0, hole=(48, 49), board=(),
             players=(Player(0, 1990, 10, 10), Player(1, 1980, 20, 20)),
             pot=30, big_blind=20, to_call=10, min_raise_to=40, max_raise_to=2000,
             can_raise=True)
    d.update(kw)
    return State(**d)


class CardsTest(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(card("As"), 51)
        self.assertEqual(card("10d"), card("Td"))
        for value in (52, -1, "XX"):
            with self.assertRaises(ValueError):
                card(value)

    def test_known_categories(self):
        hands = ["As Kd 9h 7c 4s", "As Ad Kh 7c 4s", "As Ad Kh Kc 4s",
                 "As Ad Ah Kc 4s", "As 2d 3h 4c 5s", "As Js 8s 4s 2s",
                 "As Ad Ah Kc Ks", "As Ad Ah Ac Ks", "9s Ts Js Qs Ks"]
        for category, text in enumerate(hands):
            self.assertEqual(evaluate(tuple(map(card, text.split()))) >> 20, category)

    def test_wheel_and_double_trips(self):
        wheel = tuple(map(card, "As 2d 3h 4c 5s Kd Qc".split()))
        self.assertEqual(evaluate(wheel), pack_reference((4, 5)))
        full = tuple(map(card, "As Ad Ah Kc Ks Kd 2h".split()))
        self.assertEqual(evaluate(full), pack_reference((6, 14, 13)))

    def test_partial_board_quads_all_ranks(self):
        rng = random.Random(4096)
        for r in range(13):
            board = tuple(range(r*4, r*4+4))
            self.assertEqual(evaluate(board), pack_reference((7, r+2, 0)))
            deck = [c for c in range(52) if c not in board]
            for _ in range(50):
                hole = tuple(rng.sample(deck, 2))
                x = features(hole, board)
                self.assertEqual(x[7], 7)
                self.assertEqual(x[8], r+2)
                s = make_state(hole=hole, board=board, to_call=0)
                a = Agent().decide(s)
                self.assertEqual(a, legalize(s, a))

    def test_independent_reference_6000_hands(self):
        rng = random.Random(435)
        for i in range(6000):
            n = 5 + i % 3
            cards = rng.sample(range(52), n)
            expected = max(pack_reference(reference5(c)) for c in itertools.combinations(cards, 5))
            self.assertEqual(evaluate(cards), expected, cards)

    def test_suit_permutation_invariance(self):
        rng = random.Random(58)
        for _ in range(500):
            cs = rng.sample(range(52), 7)
            perm = rng.sample(range(4), 4)
            other = [(c & ~3) + perm[c & 3] for c in cs]
            self.assertEqual(evaluate(cs), evaluate(other))
            self.assertEqual(features(cs[:2], cs[2:]), features(other[:2], other[2:]))

    def test_cpp_python_parity(self):
        exe = ROOT / "tools/equity.exe"
        if not exe.exists():
            self.skipTest("compile tools/equity.cpp for offline parity check")
        rng = random.Random(454)
        cases = [rng.sample(range(52), 5 + i % 3) for i in range(2000)]
        for r in range(13):
            board = list(range(r*4, r*4+4))
            deck = [c for c in range(52) if c not in board]
            for _ in range(20):
                cases.append(rng.sample(deck, 2) + board)
        text = "".join(f"{cs[0]} {cs[1]} {len(cs)-2} " + " ".join(map(str, cs[2:])) + "\n" for cs in cases)
        result = subprocess.run([str(exe), "features"], input=text, text=True, capture_output=True, check=True)
        for cs, line in zip(cases, result.stdout.splitlines()):
            values = list(map(int, line.split()))
            self.assertEqual(features(cs[:2], cs[2:]), values[:-1])
            self.assertEqual(evaluate(cs), values[-1])
        self.assertEqual(len(result.stdout.splitlines()), len(cases))

    def test_preflop_classes_and_blockers(self):
        counts = Counter(preflop_index(cs) for cs in itertools.combinations(range(52), 2))
        self.assertEqual(len(counts), 169)
        self.assertEqual(sum(counts.values()), 1326)
        self.assertEqual(counts[168], 6)
        self.assertGreater(PREFLOP["random"][168], .84)
        self.assertLess(PREFLOP["random"][preflop_index((20, 1))], .37)


class ModelTest(unittest.TestCase):
    def test_generated_model_matches_asset_hash(self):
        from bot.compiled_model import MODEL_SHA256
        self.assertEqual(MODEL_SHA256, hashlib.sha256((ROOT / 'assets/equity.json').read_bytes()).hexdigest())

    def test_compiled_quantization(self):
        rng = random.Random(20261106)
        for i in range(2500):
            cs = rng.sample(range(52), 5 + i % 3)
            x = features(cs[:2], cs[2:])
            for a, b in zip(predict(x), predict_reference(x)):
                self.assertLess(abs(a-b), 0.00001)

    def test_offline_prediction_parity(self):
        path = ROOT / "evidence/model_holdout.jsonl"
        if not path.exists():
            self.skipTest("holdout examples are only in the development bundle")
        for line in path.read_text().splitlines():
            case = json.loads(line)
            for actual, expected in zip(predict(case["x"]), case["prediction"]):
                self.assertAlmostEqual(actual, min(1., max(0., expected)), places=5)

    def test_prediction_bounds(self):
        rng = random.Random(55)
        for i in range(1000):
            cs = rng.sample(range(52), 5 + i % 3)
            self.assertTrue(all(0 <= x <= 1 for x in predict(features(cs[:2], cs[2:]))))


class LegalTest(unittest.TestCase):
    def test_free_check_and_call(self):
        s = make_state(to_call=0)
        self.assertEqual(legalize(s, Action("fold")), Action("check"))
        self.assertEqual(legalize(make_state(), Action("check")), Action("call"))

    def test_short_allin(self):
        s = make_state(players=(Player(0, 20, 10, 10), Player(1, 1980, 20, 20)), max_raise_to=30)
        self.assertEqual(legalize(s, Action("raise", 2000)), Action("raise", 30))

    def test_raise_not_reopened(self):
        s = make_state(can_raise=False)
        self.assertEqual(legalize(s, Action("raise", 2000)), Action("call"))

    def test_clamps_and_caps(self):
        self.assertEqual(legalize(make_state(), Action("raise", 35)), Action("raise", 40))
        self.assertEqual(legalize(make_state(), Action("raise", 9000)), Action("raise", 2000))
        s = make_state(max_raise_to=35)
        self.assertEqual(legalize(s, Action("raise", 35)), Action("call"))

    def test_validation(self):
        validate_state(make_state())
        with self.assertRaises(ValueError):
            validate_state(make_state(board=(48, 0, 4)))
        with self.assertRaises(ValueError):
            validate_state(make_state(payouts=(0, 1)))

    def test_random_state_legality_5000(self):
        rng = random.Random(2026)
        bot = Agent()
        for i in range(5000):
            cs = rng.sample(range(52), 7)
            n = (0, 3, 4, 5)[i % 4]
            stack = rng.randint(1, 10000)
            own = rng.randint(0, 400)
            current = own + rng.randint(0, 800)
            ps = (Player(0, stack, own, own), Player(1, rng.randint(0, 10000), current, current))
            s = make_state(hand_id=i, players=ps, hole=tuple(cs[:2]), board=tuple(cs[2:n+2]),
                           pot=own + current + rng.randint(0, 1000), to_call=current-own,
                           min_raise_to=current + rng.randint(20, 500), max_raise_to=own+stack,
                           can_raise=bool(i % 3))
            a = bot.decide(s)
            self.assertEqual(a, legalize(s, a))


class StrategyTest(unittest.TestCase):
    def test_aces_open(self):
        self.assertEqual(Agent().decide(make_state()).action, "raise")

    def test_check_available(self):
        self.assertNotEqual(Agent().decide(make_state(hole=(20, 1), to_call=0)).action, "fold")

    def test_table_reset(self):
        a = Agent()
        a.decide(make_state(history=(Event(1, "preflop", "raise", 60),)))
        a.decide(make_state(table_id="new", hand_id=1))
        self.assertEqual(a.stats[1].raises, 0)
        self.assertEqual(a.stats[1].hands, 1)

    def test_history_is_idempotent(self):
        a = Agent()
        s = make_state(history=(Event(1, "preflop", "raise", 60),))
        a.observe(s)
        a.observe(s)
        self.assertEqual(a.stats[1].raises, 1)
        self.assertEqual(a.stats[1].hands, 1)

    def test_public_royal_does_not_fold_profitable_split(self):
        s = make_state(hole=(0, 5), board=tuple(map(card, "Ts Js Qs Ks As".split())), pot=4000, to_call=2000)
        self.assertEqual(Agent().decide(s).action, "call")

    def test_private_royal_value(self):
        s = make_state(hole=tuple(map(card, "As Ks".split())), board=tuple(map(card, "Ts Js Qs 2h 3d".split())), pot=400, to_call=0)
        self.assertEqual(Agent().decide(s).action, "raise")

    def test_short_allin_does_not_count_ineligible_sidepot(self):
        # Hero can win only 150 chips from each of three players. A very
        # large side pot must not turn a weak hand into a cheap call.
        board = tuple(map(card, "Ah Kd Qc 8s 4h".split()))
        ps = (Player(0, 50, 0, 100), Player(1, 2000, 1000, 1100), Player(2, 2000, 1000, 1100))
        s = make_state(hole=tuple(map(card, "3c 2d".split())), board=board, players=ps,
                       pot=2300, to_call=1000, min_raise_to=2000, max_raise_to=50)
        self.assertEqual(Agent().decide(s).action, "fold")

    def test_public_quads_with_nut_kicker_never_fold(self):
        for rank in range(13):
            board = tuple(range(rank*4, rank*4+4))
            high = 44 if rank == 12 else 48
            low = next(c for c in range(52) if c not in board and c != high)
            s = make_state(hole=(high, low), board=board, pot=4000, to_call=1990)
            self.assertEqual(Agent().decide(s).action, 'call')


class ICMTest(unittest.TestCase):
    def test_winner_take_all(self):
        result = icm((100, 200, 300), (1, 0, 0))
        for a, b in zip(result, (1/6, 2/6, 3/6)):
            self.assertAlmostEqual(a, b)

    def test_equal_stacks_and_conservation(self):
        for n in range(2, 9):
            values = icm((2000,) * n, (.5, .3, .2))
            self.assertAlmostEqual(sum(values), sum((.5, .3, .2)[:n]))
            self.assertLess(max(values) - min(values), 1e-12)

    def test_risk_optional(self):
        p = (Player(0, 1000), Player(1, 1000), Player(2, 300))
        self.assertEqual(risk_premium(p, 0, (), 1), 0)
        self.assertEqual(risk_premium(p, 0, (1, 0, 0), 1), 0)
        self.assertGreater(risk_premium(p, 0, (.5, .3, .2), 1), 0)


class SimulatorTest(unittest.TestCase):
    def _reopening_scenario(self, second_short):
        class Script:
            def __init__(self, moves):
                self.moves = list(moves)
                self.states = []
            def decide(self, state):
                self.states.append(state)
                return self.moves.pop(0) if self.moves else Action("call" if state.to_call else "check")
        scripts = [Script([Action("raise", 100)]), Script([Action("raise", 140)]),
                   Script([Action("raise", 180)] if second_short else [Action("call")]), Script([Action("call")])]
        table = Table(scripts, stacks=[1000, 140, 180 if second_short else 1000, 1000])
        table.dealt = list(range(4))
        table.board = []
        table.history = []
        for i, p in enumerate(table.seats):
            p.hole = (i*2, i*2+1)
        table.betting(20, 0, True)
        return scripts[0].states[-1]

    def test_single_short_raise_does_not_reopen(self):
        state = self._reopening_scenario(False)
        self.assertEqual(state.to_call, 40)
        self.assertFalse(state.can_raise)

    def test_cumulative_short_raises_reopen(self):
        state = self._reopening_scenario(True)
        self.assertEqual(state.to_call, 80)
        self.assertTrue(state.can_raise)
        self.assertEqual(state.min_raise_to, 260)

    def test_side_pots(self):
        self.assertEqual(distribute([100, 300, 300], [False]*3, [30, 20, 10], [0, 1, 2]), [300, 400, 0])
        self.assertEqual(distribute([100, 300, 500], [False]*3, [30, 20, 10], [0, 1, 2]), [300, 400, 200])

    def test_folded_dead_money_and_odd_chips(self):
        self.assertEqual(distribute([101, 101, 101], [False, False, True], [20, 20, 30], [1, 2, 0]), [151, 152, 0])

    def test_short_stacks_conserve(self):
        table = Table([Agent(i+1) for i in range(8)], seed=18, stacks=[1, 5, 10, 15, 20, 40, 100, 500], payouts=(.5, .3, .2), hand_cap=30)
        table.run()
        self.assertEqual(sum(p.stack for p in table.seats), 691)

    def test_heads_up_rotation(self):
        table = Table([Baseline("equity", 1), Agent()], seed=2, payouts=(1, 0), hand_cap=25)
        table.run()
        self.assertEqual(sum(p.stack for p in table.seats), 4000)

    def test_eight_seat_tournaments(self):
        for seed in range(4):
            table = Table([Agent(i + seed * 8 + 1) for i in range(8)], seed=seed, payouts=(.5, .3, .2), hand_cap=40)
            table.run()
            self.assertEqual(sum(p.stack for p in table.seats), 16000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
