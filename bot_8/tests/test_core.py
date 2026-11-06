import itertools
import random
import unittest
from collections import Counter
from dataclasses import replace

from agent import Agent, Action, Event, Observation, Player
from kestrel.cards import card, text, evaluate, hand_class, class_name
from kestrel.equity import preflop_equity, estimate, fast_quality, made_quality, board_context, POOLS
from kestrel.icm import value, call_threshold
from kestrel.model import legalize, contestable_pots
from tools.simulator import Table, Seat, split_pots, action_is_legal
from tools.baselines import Baseline


def independent_five(cards):
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    counts = Counter(ranks)
    groups = sorted(((count, r) for r, count in counts.items()), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    unique = sorted(counts, reverse=True)
    high = 5 if unique == [14, 5, 4, 3, 2] else unique[0]
    straight = len(unique) == 5 and (unique[0] - unique[-1] == 4 or high == 5)
    if straight and flush:
        cat, kickers = 8, [high]
    elif groups[0][0] == 4:
        cat, kickers = 7, [groups[0][1], groups[1][1]]
    elif [g[0] for g in groups] == [3, 2]:
        cat, kickers = 6, [groups[0][1], groups[1][1]]
    elif flush:
        cat, kickers = 5, ranks
    elif straight:
        cat, kickers = 4, [high]
    elif groups[0][0] == 3:
        cat, kickers = 3, [groups[0][1]] + sorted((r for r in ranks if r != groups[0][1]), reverse=True)
    elif groups[0][0] == 2 and groups[1][0] == 2:
        cat, kickers = 2, [max(groups[0][1], groups[1][1]), min(groups[0][1], groups[1][1]), groups[2][1]]
    elif groups[0][0] == 2:
        cat, kickers = 1, [groups[0][1]] + sorted((r for r in ranks if r != groups[0][1]), reverse=True)
    else:
        cat, kickers = 0, ranks
    return (cat << 20) | sum(r << (16 - 4 * i) for i, r in enumerate(kickers))


def obs(**changes):
    x = Observation(1, 0, 1, (card("As"), card("Kd")), (),
                    (Player(0, 1980, 20, 20), Player(1, 1960, 40, 40)),
                    60, 20, 60, 2000, 20)
    return replace(x, **changes)


class CardsTests(unittest.TestCase):
    def test_card_roundtrip(self):
        for c in range(52):
            self.assertEqual(card(text(c)), c)
        self.assertEqual(card("10H"), card("Th"))
        for bad in [52, -1, True, "AA", "1s", "Asx", None]:
            with self.assertRaises(ValueError):
                card(bad)

    def test_categories_and_order(self):
        examples = ["As Kd Qc 9h 7s", "As Ad Qc 9h 7s", "As Ad Qc Qh 7s",
                    "As Ad Ac 9h 7s", "As 2d 3c 4h 5s", "As Js 8s 4s 2s",
                    "As Ad Ac 9h 9s", "As Ad Ac Ah 7s", "As Ks Qs Js Ts"]
        values = [evaluate(tuple(map(card, e.split()))) for e in examples]
        self.assertEqual([v >> 20 for v in values], list(range(9)))
        self.assertEqual(sorted(values), values)

    def test_seven_against_independent_oracle(self):
        rng = random.Random(8128)
        for size in (5, 6, 7):
            for _ in range(1200):
                hand = rng.sample(range(52), size)
                self.assertEqual(evaluate(hand), max(independent_five(c) for c in itertools.combinations(hand, 5)), hand)

    def test_two_trips_and_board_ties(self):
        hand = tuple(map(card, "As Ah Ad Ks Kh Kd 2h".split()))
        self.assertEqual(evaluate(hand), independent_five(tuple(map(card, "As Ah Ad Ks Kh".split()))))
        board = tuple(map(card, "As Ks Qs Js Ts".split()))
        self.assertEqual(evaluate(board + (card("2h"), card("3h"))), evaluate(board + (card("7d"), card("8d"))))

    def test_classes(self):
        self.assertEqual(len({hand_class(a, b) for a in range(52) for b in range(a + 1, 52)}), 169)
        self.assertEqual(class_name(hand_class(card("As"), card("Ks"))), "AKs")
        self.assertEqual(class_name(hand_class(card("As"), card("Kd"))), "AKo")
        self.assertGreater(preflop_equity(hand_class(card("As"), card("Ad")), 1), .84)

    def test_range_pools_preserve_suit_symmetry(self):
        for pool in POOLS:
            counts=Counter(hand_class(*h) for h in pool)
            for cls,n in counts.items():
                a,b=divmod(cls,13)
                self.assertEqual(n,6 if a==b else 4 if a<b else 12)

    def test_cached_board_quality_equivalence(self):
        rng=random.Random(818)
        for n in (3,4,5):
            for _ in range(300):
                cards=rng.sample(range(52),n+2)
                hole,board=tuple(cards[:2]),tuple(cards[2:])
                v=evaluate(cards)
                self.assertEqual(made_quality(hole,board,v),fast_quality(hole,board_context(board),v))


class ICMTests(unittest.TestCase):
    def test_symmetry_conservation_scaling(self):
        pay = (8., 5., 3., 0.)
        self.assertAlmostEqual(value([100]*4, 0, pay), 4.0)
        stacks = [17, 39, 130, 61]
        self.assertAlmostEqual(sum(value(stacks, h, pay) for h in range(4)), sum(pay))
        self.assertAlmostEqual(value(stacks, 1, pay), value([s * 100 for s in stacks], 1, pay))

    def test_against_permutations(self):
        stacks = [21, 37, 86, 19, 4]
        pay = [10, 5, 3, 1, 0]
        exact = [0.] * 5
        for order in itertools.permutations(range(5)):
            prob, remaining = 1., sum(stacks)
            for j in order:
                prob *= stacks[j] / remaining
                remaining -= stacks[j]
            for place, j in enumerate(order):
                exact[j] += prob * pay[place]
        for j in range(5):
            self.assertAlmostEqual(value(stacks, j, pay), exact[j])

    def test_winner_take_all_and_bubble(self):
        self.assertAlmostEqual(value([100, 200, 300], 1, [1, 0, 0]), 1/3)
        x = obs(players=(Player(0, 100, 0, 0), Player(1, 0, 100, 100), Player(2, 20, 0, 0, True)),
                pot=100, to_call=100, max_raise_to=100, payouts=(1., 1., 0.))
        self.assertGreater(call_threshold(x, 100, 1), .8)


class InterfaceTests(unittest.TestCase):
    def test_validation(self):
        obs().validate()
        for x in [obs(hole=(1,1)), obs(board=(3,4)), obs(to_call=-1), obs(payouts=(0.,1.)), obs(max_raise_to=9999)]:
            with self.assertRaises(ValueError):
                x.validate()

    def test_raise_gate(self):
        self.assertEqual(legalize(obs(), Action("raise", 10)), Action("raise", 60))
        self.assertEqual(legalize(obs(can_raise=False), Action("raise", 500)), Action("call"))
        x = obs(players=(Player(0,30,20,20), Player(1,1960,40,40)), max_raise_to=50)
        self.assertEqual(legalize(x, Action("raise", 100)), Action("raise", 50))
        self.assertEqual(legalize(obs(to_call=0), Action("fold")), Action("check"))

    def test_state_reset_and_dedup(self):
        bot = Agent()
        x = obs(history=(Event(0,1,"preflop","raise",40,30,10),))
        bot.act(x)
        bot.act(x)
        self.assertEqual(bot.stats[1].raises, 1)
        bot.act(replace(x, table_id="new", history=()))
        self.assertEqual(bot.stats, {})

    def test_equity_nuts_and_board_ties(self):
        board = tuple(map(card, "As Ks Qs Js Ts".split()))
        for n in [1,3,7]:
            q, _, _ = estimate((card("2h"),card("3h")), board, [(7,0,0,.1)]*n, 8)
            self.assertAlmostEqual(q, 1/(n+1))
        board = tuple(map(card, "Qs Js Ts 2h 3d".split()))
        q, _, _ = estimate((card("As"),card("Ks")),board,[(7,2,0,.1)],8)
        self.assertEqual(q, 1.0)

    def test_no_fold_free_option(self):
        bot = Agent()
        for h in [(0,5), (48,49), (12,25)]:
            x = obs(hole=h, to_call=0)
            action = bot.act(x)
            self.assertNotEqual(action.action,"fold")
            self.assertTrue(action_is_legal(x,action))

    def test_short_caller_cannot_count_unwinnable_chips(self):
        x=obs(players=(Player(0,50,50,50),Player(1,0,1000,1000,all_in=True)),pot=1050,to_call=950,max_raise_to=100)
        eligible,pots=contestable_pots(x,50,[x.players[1]])
        self.assertEqual(eligible,150)
        self.assertIsNone(pots)
        # Actual call price is 50/(150+50)=25%, not 50/1100.
        self.assertEqual(50/(eligible+50),.25)

    def test_covering_player_can_win_only_side_pot(self):
        x=obs(players=(Player(0,500,500,500),Player(1,0,100,100,all_in=True),Player(2,500,500,500)),pot=1100,to_call=0,max_raise_to=1000)
        eligible,pots=contestable_pots(x,0,list(x.players[1:]))
        self.assertEqual(eligible,1100)
        self.assertEqual(pots,((300/1100,(0,1)),(800/1100,(1,))))

    def test_exact_replay_with_budget_disabled(self):
        x=obs(board=(card('Qh'),card('Jh'),card('2c')))
        a=Agent(time_budget_us=None)
        b=Agent(time_budget_us=None)
        self.assertEqual(a.act(x),b.act(x))
        self.assertEqual(a.diagnostics,b.diagnostics)


class Scripted:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.observations = []
    def act(self, o):
        self.observations.append(o)
        return next(self.actions)


class SimulatorTests(unittest.TestCase):
    def test_side_pots(self):
        won = split_pots({0:100,1:200,2:300},set(),{0:30,1:20,2:10},0,[0,1,2])
        self.assertEqual(won,{0:300,1:200,2:100})
        won = split_pots({0:1,1:1,2:1},{2},{0:10,1:10},0,[0,1,2])
        self.assertEqual(won,{0:1,1:2,2:0})

    def _round_table(self, stacks, scripts):
        agents = [Scripted(s) for s in scripts]
        t = Table(agents,stacks=stacks,points=tuple(range(len(stacks),0,-1)))
        t.alive = list(range(len(stacks)))
        t.button = len(stacks)-1
        t.bb=20
        t.holes = {s:(2*s,2*s+1) for s in t.alive}
        t.board = [40,41,42]
        return t, agents

    def test_short_raise_does_not_reopen(self):
        t,a = self._round_table([1000,150,1000],[
            [Action("raise",100),Action("call")], [Action("raise",150)], [Action("call")]])
        t._round(0,[])
        self.assertFalse(a[0].observations[-1].can_raise)
        self.assertEqual(a[0].observations[-1].to_call,50)

    def test_cumulative_short_raises_reopen(self):
        t,a = self._round_table([1000,140,200,1000],[
            [Action("raise",100),Action("raise",300)], [Action("raise",140)],
            [Action("raise",200)], [Action("call"),Action("call")]])
        t._round(0,[])
        self.assertTrue(a[0].observations[-1].can_raise)
        self.assertEqual(a[0].observations[-1].min_raise_to,300)

    def test_headsup_button_acts_first_preflop(self):
        captured=[]
        t=Table([Baseline(),Baseline()],seed=3,hand_cap=1,capture=lambda o,a:captured.append(o))
        t.run()
        self.assertEqual(captured[0].hero,captured[0].button)
        self.assertEqual(captured[0].to_call,10)

    def test_short_big_blind_no_phantom_call(self):
        t=Table([Baseline("station"),Baseline("station")],seed=2,stacks=[2000,5],hand_cap=2,points=[1,0])
        result=t.run()
        self.assertEqual(sum(result["stacks"]),2005)

    def test_random_tournaments_conserve_chips_and_points(self):
        decisions=0
        for seed in range(20):
            t=Table([Agent()]+[Baseline(s) for s in ["abc","nit","lag","station","random","abc","lag"]],seed=seed)
            result=t.run()
            self.assertEqual(sum(result["stacks"]),16000)
            self.assertAlmostEqual(sum(result["points"]),19)
            decisions+=result["actions"]
        self.assertGreater(decisions,4000)


if __name__ == "__main__":
    unittest.main()
