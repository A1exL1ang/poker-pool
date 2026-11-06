"""Policy regressions against independently constructed Hold'em situations."""
import random
import unittest

from agent import Agent
from cards import parse_card
from engine import Hand, play_hand
from opponents import CheckCallBot, ProfileBot


def cs(text):
    return [parse_card(c) for c in text.split()]


def river_observation(hole, board, call=0):
    h = Hand([CheckCallBot()] * 2, [1000, 1000], button=0)
    h.holes[0], h.board = cs(hole), cs(board)
    h.street = 3
    for p in h.players:
        p.update(stack=900, bet=0, committed=100, all_in=False)
    h.maxbet, h.last_full_raise = 0, 20
    h.last_action_bet = {}
    if call:
        h.apply_action(1, dict(action='raise', amount=call))
    return h.observation(0)


def risk_observation(payouts, zero_seat=False):
    # Hero has 20 already in and must call 80 against a villain with 100 in.
    # Third player folded, retaining100; endpoints sum to300 chips.
    players = [dict(seat=0, stack=80, bet=20, committed=20, folded=False, all_in=False),
               dict(seat=1, stack=0, bet=100, committed=100, folded=False, all_in=True),
               dict(seat=2, stack=100, bet=0, committed=0, folded=True, all_in=False)]
    if zero_seat:
        players.append(dict(seat=3, stack=0, bet=0, committed=0, folded=True, all_in=False))
    return dict(players=players, hero=0, pot=120, to_call=80, payouts=payouts)


class AgentTests(unittest.TestCase):
    def test_allin_blind_aces_call(self):
        h = Hand([CheckCallBot()] * 2, [1000, 20], button=0)
        h.holes[0] = cs('Ac Ad')
        o = h.observation(0)
        self.assertFalse(o['can_raise'])
        self.assertEqual(o['to_call'], 10)
        self.assertEqual(Agent().act(o), dict(action='call'))

    def test_short_call_eligible_pot(self):
        h = Hand([CheckCallBot()] * 2, [500, 100], button=0)
        h.apply_action(0, dict(action='raise', amount=500))
        o = h.observation(1)
        self.assertEqual(o['pot'], 520)
        self.assertEqual(Agent._eligible_pot(o), 120)
        self.assertEqual(Agent._eligible_pot(o) + o['to_call'], 200)

    def test_shared_royal_never_folds(self):
        o = river_observation('2d 3h', 'Tc Jc Qc Kc Ac', call=900)
        self.assertEqual(Agent().act(o), dict(action='call'))
        o = river_observation('2d 3h', 'Tc Jc Qc Kc Ac')
        self.assertEqual(Agent().act(o), dict(action='check'))

    def test_shared_broadway_never_folds(self):
        o = river_observation('2d 3h', 'Tc Jd Qh Ks Ac', call=900)
        self.assertEqual(Agent().act(o), dict(action='call'))

    def test_shared_quad_ace_kicker_never_folds(self):
        o = river_observation('3d 4h', '2c 2d 2h 2s Ac', call=900)
        self.assertEqual(Agent().act(o), dict(action='call'))
        # All aces on the board makes a king the highest possible kicker.
        o = river_observation('2d 3h', 'Ac Ad Ah As Kc', call=900)
        self.assertEqual(Agent().act(o), dict(action='call'))

    def test_own_royal_jams_when_allowed(self):
        for call in [0, 100]:
            o = river_observation('Ac Kc', 'Qc Jc Tc 2d 3h', call=call)
            a = Agent().act(o)
            self.assertTrue(o['can_raise'])
            self.assertEqual(a, dict(action='raise', amount=o['max_raise_to']))

    def test_icm_heads_up_is_chip_odds(self):
        # Fold: stacks80/120 -> utility52. Win:200/0 ->100.
        # Lose:0/200 ->20. Therefore(52-20)/(100-20)=.4.
        o = risk_observation([100, 20])
        o['players'] = o['players'][:2]
        a = Agent()
        self.assertAlmostEqual(a._risk_threshold(o, [o['players'][1]], .4), .4)

    def test_icm_three_survivor_hand_derived_threshold(self):
        # Fold hero first prob4/15, second prob14/45.
        # For100/60/0: fold136/3, win260/3, lose0 =>34/65.
        a = Agent()
        o = risk_observation([100, 60, 0])
        self.assertAlmostEqual(a._risk_threshold(o, [o['players'][1]], .4), 34 / 65)
        # For100/60/20: fold484/9, win260/3, lose20 =>38/75.
        # Zero-stack original seats do not consume a remaining finishing place.
        o = risk_observation([100, 60, 20], zero_seat=True)
        self.assertAlmostEqual(a._risk_threshold(o, [o['players'][1]], .4), 38 / 75)

    def test_icm_unmatched_overbet_refunded_in_win_state(self):
        # Hero can win200 total; villain retains400 even if hero wins.
        # A positive second prize makes the risk premium independently testable.
        h = Hand([CheckCallBot()] * 3, [100, 500, 200], button=0)
        h.players = [dict(seat=0, stack=80, bet=20, committed=20, folded=False, all_in=False),
                     dict(seat=1, stack=0, bet=500, committed=500, folded=False, all_in=True),
                     dict(seat=2, stack=200, bet=0, committed=0, folded=True, all_in=False)]
        h.maxbet = 500
        o = h.observation(0)
        o['payouts'] = [100, 60, 20]
        # Direct three-player marginal finish probabilities, independent of DP.
        def three_player_hero(stacks):
            a, b, c = stacks
            total = a + b + c
            first = a / total
            second = b / total * a / (a + c) + c / total * a / (a + b)
            return first * 100 + second * 60 + (1 - first - second) * 20
        uf = three_player_hero([80, 520, 200])
        uw = three_player_hero([200, 400, 200])
        ul = 20
        expected = max(.4, min(.95, (uf - ul) / (uw - ul)))
        self.assertAlmostEqual(Agent()._risk_threshold(o, [o['players'][1]], .4), expected)

    def test_duplicate_invalid_cards_fallback(self):
        invalid = [([0, 0], []), ([0, 1], [0, 4, 8]), ([0, 52], []), ([-1, 3], []),
                   ([True, 3], []), (['Ac', 3], []), ([0], []), ([0, 1], [2])]
        for hole, board in invalid:
            for due in [0, 10]:
                o = dict(hole=hole, board=board, to_call=due)
                a = Agent()
                self.assertEqual(a.act(o), dict(action='fold' if due else 'check'))
                self.assertEqual(a.failures, 1)

    def test_preflop_stats_count_once_per_hand(self):
        a = Agent()
        h = Hand([CheckCallBot()] * 3, [2000] * 3, button=0)
        o = h.observation(0)
        o['history'] += [dict(seat=1, street=0, action='call', amount=10),
                         dict(seat=1, street=0, action='raise', amount=60),
                         dict(seat=1, street=0, action='raise', amount=180)]
        a._observe(o)
        self.assertEqual(a.stats[1][1:3], [1, 1])

    def test_unobserved_actions_do_not_make_opponents_look_tight(self):
        a=Agent()
        h=Hand([CheckCallBot()]*3,[2000]*3,button=0)
        for hand in range(100):
            o=h.observation(0);o['hand_id']=hand
            a._observe(o)
        self.assertEqual(a.stats[1][0],0)
        self.assertAlmostEqual(a._style(1)[1],.18)
        o['history'] += [dict(seat=1,street=0,action='call',amount=10),
                         dict(seat=1,street=0,action='raise',amount=60)]
        a._observe(o)
        self.assertEqual(a.stats[1][:3],[1,1,1])
        a._observe(o)
        self.assertEqual(a.stats[1][1:3], [1, 1])

    def test_ten_thousand_collected_actions_are_legal(self):
        rng = random.Random(584731)
        collected = [0]
        class Checked:
            def __init__(wrapper, policy):
                wrapper.policy = policy
            def act(wrapper, o):
                action = wrapper.policy.act(o)
                collected[0] += 1
                self.assertEqual(wrapper.policy.failures, 0)
                self.assertIn(action['action'], ['fold', 'check', 'call', 'raise'])
                if action['action'] == 'check':
                    self.assertEqual(o['to_call'], 0)
                if action['action'] == 'raise':
                    self.assertTrue(o['can_raise'])
                    self.assertIs(type(action['amount']), int)
                    self.assertLessEqual(action['amount'], o['max_raise_to'])
                    self.assertTrue(action['amount'] >= o['min_raise_to'] or action['amount'] == o['max_raise_to'])
                return action
        for seed in range(10000):
            n = rng.randint(2, 8)
            bb = rng.choice([20, 40, 100, 200, 800, 1600])
            stacks = [rng.randint(1, 4000) for _ in range(n)]
            bots = [Checked(Agent(seed * 10 + i, {'pushfold': bool(seed % 2)})) if i % 2 == 0 else
                    ProfileBot(rng.choice(['random', 'station', 'tight', 'maniac']), seed * 10 + i)
                    for i in range(n)]
            r = play_hand(bots, stacks, button=seed % n, small_blind=bb // 2,
                          big_blind=bb, seed=seed, hand_id=seed)
            self.assertFalse(r['errors'], (seed, r['errors']))
            if collected[0] >= 10000:
                break
        self.assertGreaterEqual(collected[0], 10000)


if __name__ == '__main__':
    unittest.main()
