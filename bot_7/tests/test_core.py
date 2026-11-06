import itertools
import random
import unittest
from bot7.cards import card, hand_class, name
from bot7.evaluator import evaluate
from bot7.equity import equity, preflop_equity
from bot7.icm import utility, call_threshold
from bot7.interface import Action, Observation, Player, legalize
from bot7.agent import Agent
from tools.simulator import Table, settle
from tools.baselines import Baseline


def reference_five(cards):
    """Independent slow evaluator: tuple ordering, no generated lookup."""
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    groups = sorted(((ranks.count(r), r) for r in set(ranks)), reverse=True)
    unique = sorted(set(ranks), reverse=True)
    straight = unique[0] if len(unique) == 5 and unique[0] - unique[-1] == 4 else 5 if unique == [14, 5, 4, 3, 2] else 0
    flush = len({c % 4 for c in cards}) == 1
    if straight and flush:
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
        return (3, groups[0][1], *sorted((r for n, r in groups if n == 1), reverse=True))
    if [g[0] for g in groups[:2]] == [2, 2]:
        return (2, max(groups[0][1], groups[1][1]), min(groups[0][1], groups[1][1]), groups[2][1])
    if groups[0][0] == 2:
        return (1, groups[0][1], *sorted((r for n, r in groups if n == 1), reverse=True))
    return (0, *ranks)


def packed(value):
    return (value[0] << 20) + sum(r << (16 - 4 * i) for i, r in enumerate(value[1:]))


class EvaluationTests(unittest.TestCase):
    def test_cards(self):
        self.assertEqual([card(name(c)) for c in range(52)], list(range(52)))
        self.assertEqual(card("10s"), card("Ts"))
        self.assertEqual(hand_class(card("Ac"), card("Kc")), "AKs")

    def test_random_reference(self):
        rng = random.Random(47)
        for n in (5, 6, 7):
            for _ in range(2500):
                hand = rng.sample(range(52), n)
                expected = max(reference_five(c) for c in itertools.combinations(hand, 5))
                self.assertEqual(evaluate(hand), packed(expected), hand)

    def test_pathologies(self):
        cases = ["As 2s 3s 4s 5s Kh Kd", "Ac Ad Ah Kc Kd Kh 2s", "Ac Ad Ah As Kd Kc 2s",
                 "2c 3c 4c 5c 7c 6d 6h", "Ah Kh Qh Jh 9h Th 2h", "As Ad Ks Kd Qs Qd Jh"]
        for text in cases:
            hand = list(map(card, text.split()))
            self.assertEqual(evaluate(hand), packed(max(reference_five(c) for c in itertools.combinations(hand, 5))))

    def test_preflop(self):
        self.assertTrue(.84 < preflop_equity(card("As"), card("Ah")) < .87)
        self.assertGreater(preflop_equity(card("As"), card("Ah"), .05), .78)

    def test_board_ties(self):
        board = tuple(map(card, "As Ks Qs Js Ts".split()))
        for n in (1, 2, 7):
            value = equity((card("2c"), card("3d")), board, [(1, 0)] * n, random.Random(1))
            self.assertAlmostEqual(value, 1 / (n + 1), places=10)

    def test_weighted_sidepot_equity(self):
        board = tuple(map(card, "As Ks Qs Js Ts".split()))
        value = equity((card("2c"), card("3d")), board, [(1, 0)] * 2, random.Random(1),
                       pot_groups=[(60, (0, 1)), (160, (1,))])
        self.assertAlmostEqual(value, (60 / 3 + 160 / 2) / 220, places=10)

    def test_pot_groups(self):
        me = Player(0, 100, 0, 60)
        opponents = (Player(1, 0, 0, 20, all_in=True), Player(2, 100, 0, 100))
        obs = Observation((0, 1), (4, 5, 6), (me,) + opponents, 0, 2, 180, 40, 80, 100, ("fold", "call", "raise"))
        self.assertEqual(Agent._pot_groups(obs, me, opponents), [(60, (0, 1)), (160, (1,))])


class ICMTests(unittest.TestCase):
    def test_eight_player_dp(self):
        def dp(stacks, hero, payouts):
            others = [x for i, x in enumerate(stacks) if i != hero]
            total = sum(stacks)
            probability = [0.0] * (1 << len(others))
            probability[0] = 1
            result = 0.0
            for mask in range(len(probability)):
                remaining = total - sum(v for i, v in enumerate(others) if mask & (1 << i))
                result += probability[mask] * stacks[hero] / remaining * payouts[mask.bit_count()]
                for i, v in enumerate(others):
                    if not mask & (1 << i):
                        probability[mask | (1 << i)] += probability[mask] * v / remaining
            return result
        rng = random.Random(43)
        for _ in range(40):
            stacks = [rng.randrange(1, 20000) for _ in range(8)]
            payouts = sorted((rng.randrange(1000) for _ in range(8)), reverse=True)
            hero = rng.randrange(8)
            self.assertAlmostEqual(utility(stacks, hero, payouts), dp(stacks, hero, payouts), places=8)

    def test_brute_force(self):
        stacks = [7, 11, 19, 31, 47]
        for payouts in ([50, 30, 20, 0, 0], [8, 7, 6, 5, 4], [80, 13, 5, 2, 1]):
            values = [0.0] * len(stacks)
            for order in itertools.permutations(range(len(stacks))):
                remaining = sum(stacks)
                probability = 1.0
                for seat in order:
                    probability *= stacks[seat] / remaining
                    remaining -= stacks[seat]
                for place, seat in enumerate(order):
                    values[seat] += probability * payouts[place]
            for hero in range(len(stacks)):
                self.assertAlmostEqual(utility(stacks, hero, payouts), values[hero])

    def test_winner_take_all(self):
        stacks = [20, 30, 50]
        for i in range(3):
            self.assertAlmostEqual(utility(stacks, i, [100, 0, 0]), stacks[i])

    def test_symmetry_and_sum(self):
        payouts = [50, 30, 20, 0]
        for i in range(4):
            self.assertAlmostEqual(utility([100] * 4, i, payouts), 25)
        self.assertAlmostEqual(sum(utility([4, 8, 15, 23], i, payouts) for i in range(4)), 100)

    def test_price_and_bubble(self):
        self.assertAlmostEqual(call_threshold([80, 50, 200], 0, 1, 70, 30, [1, 0, 0]), .3)
        self.assertGreater(call_threshold([80, 0, 100], 0, 1, 100, 80, [50, 30, 0]), 80 / 180)


class EngineTests(unittest.TestCase):
    def test_no_phantom_bet_after_short_big_blind(self):
        class Caller:
            def act(self, obs):
                return Action("call" if obs.to_call else "check")
        class Folder:
            def act(self, obs):
                return Action("fold" if obs.to_call else "check")
        table = Table([Caller(), Folder(), Caller(), Caller()], stacks=[7, 20, 5, 7])
        result = table.play_hand(20)
        self.assertFalse(result.folded[1])
        self.assertGreaterEqual(table.stacks[1], 13)
        self.assertEqual(sum(table.stacks), 39)

    def test_short_call_eligible_pot(self):
        me = Player(0, 5)
        obs = Observation((0, 1), (), (me, Player(1, 90, 10, 10), Player(2, 80, 20, 20)),
                          0, 0, 30, 5, 40, 5, ("fold", "call"))
        self.assertEqual(Agent._eligible_pot(obs, me), 10)

    def test_derived_sidepot_skips_heads_up_icm(self):
        me = Player(0, 5)
        villain = Player(1, 80, 20, 20)
        obs = Observation((0, 1), (), (me, villain), 0, 0, 20, 5, 40, 5,
                          ("fold", "call"), payouts=(1, 0))
        agent = Agent()
        self.assertEqual(agent._risk(obs, me, [villain], agent._eligible_pot(obs, me)), 0)

    def test_nuts_call_allin(self):
        obs = Observation(tuple(map(card, ("As", "Ks"))), tuple(map(card, "Qs Js Ts 2d 3c".split())),
                          (Player(0, 1000, 0, 100), Player(1, 0, 1000, 1100, all_in=True)),
                          0, 1, 1200, 1000, 2000, 1000, ("fold", "call"))
        agent = Agent(budget_us=0)
        self.assertEqual(agent.act(obs), Action("call"))
        self.assertEqual(agent.errors, 0)

    def test_cumulative_short_allins_reopen(self):
        class Script:
            def __init__(self, target=None):
                self.target = target
                self.calls = []
            def act(self, obs):
                self.calls.append(obs)
                if obs.street == "preflop" and len(self.calls) == 1 and self.target is not None:
                    return Action("raise", self.target)
                return Action("call" if obs.to_call else "check")
        for small, large, reopened in ((140, 180, True), (130, 150, False)):
            agents = [Script(small), Script(large), Script(), Script(100)]
            table = Table(agents, stacks=[small, large, 2000, 2000])
            table.play_hand(20)
            second = agents[3].calls[1]
            self.assertEqual(second.street, "preflop")
            self.assertEqual("raise" in second.legal_actions, reopened)
            self.assertEqual(second.min_raise_to, large + 80)

    def test_invalid_visible_cards(self):
        obs = Observation((0, 0), (), (Player(0, 100), Player(1, 100)), 0, 1, 0, 0, 20, 100, ("check", "raise"))
        with self.assertRaises(ValueError):
            obs.validate()

    def test_sidepots(self):
        self.assertEqual(settle([100, 200, 300], [False] * 3, [9, 8, 7], 0), [300, 200, 100])
        self.assertEqual(settle([100, 200, 300], [True, False, False], [9, 8, 7], 0), [0, 500, 100])
        self.assertEqual(sum(settle([11, 11, 11], [False] * 3, [7, 7, 0], 0)), 33)

    def test_short_raise_legalizer(self):
        obs = Observation((0, 1), (), (Player(0, 5), Player(1, 10)), 0, 1, 10, 0, 20, 5, ("check", "raise"))
        self.assertEqual(legalize(obs, Action("raise", 100)), Action("raise", 5))
        self.assertEqual(legalize(obs, Action("call")), Action("check"))

    def test_random_games(self):
        for seed in range(8):
            agents = [Agent(seed=seed * 8 + i, comparisons=28) if i == 0 else Baseline(["abc", "station", "random", "nit"][i % 4], seed * 8 + i) for i in range(8)]
            table = Table(agents, seed=seed, stacks=[37, 80, 120, 450, 800, 2000, 3500, 7000])
            for h in range(50):
                table.play_hand(20 * (1 + h // 10))
                self.assertEqual(sum(table.stacks), table.initial_total)
            self.assertEqual(agents[0].errors, 0)

    def test_heads_up(self):
        table = Table([Baseline("station"), Baseline("station")], stacks=[20, 20])
        table.capture = True
        table.play_hand(20)
        first = table.observations[0][1]
        self.assertEqual(first.hero, first.button)
        self.assertEqual(first.to_call, 10)
        self.assertEqual(sum(table.stacks), 40)

    def test_table_reset(self):
        agent = Agent()
        agent.models[5] = "stale"
        obs = Observation((48, 49), (), (Player(0, 1990, 10, 10), Player(1, 1980, 20, 20)),
                          0, 0, 30, 10, 40, 2000, ("fold", "call", "raise"), table_id="new")
        agent.act(obs)
        self.assertNotIn(5, agent.models)
        self.assertEqual(agent.errors, 0)


if __name__ == "__main__":
    unittest.main()
