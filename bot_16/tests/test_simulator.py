"""Mechanics, information-boundary, and independent evaluator checks."""
import random
import unittest

from simulator import evaluate, evaluate_reference, icm_values, play_hand, play_table


def cards(text):
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    return [ranks.index(c[0]) * 4 + suits.index(c[1]) for c in text.split()]


def deck_with(holes, board, button=0):
    n = len(holes)
    order = [(button + offset) % n for offset in range(1, n + 1)]
    front = [holes[i][j] for j in range(2) for i in order] + board
    assert len(set(front)) == len(front)
    return front + [c for c in range(52) if c not in front]


class Scripted:
    def __init__(self, actions=None):
        self.actions = actions or {}
        self.seen = []
        self.counts = {}

    def act(self, obs):
        street = obs['street']
        count = self.counts.get(street, 0)
        self.counts[street] = count + 1
        self.seen.append(obs)
        action = self.actions.get((street, count))
        if callable(action):
            return action(obs)
        if action is not None:
            return action
        return {'action': 'call' if obs['to_call'] else 'check'}


def raise_to(amount):
    return {'action': 'raise', 'amount': amount}


def shove(obs):
    if obs['can_raise']:
        return raise_to(obs['max_raise_to'])
    return {'action': 'call' if obs['to_call'] else 'check'}


class MechanicsTests(unittest.TestCase):
    def test_heads_up_blinds_and_order(self):
        agents = [Scripted(), Scripted()]
        result = play_hand(agents, [1000, 1000], button=0)
        self.assertEqual([(a['seat'], a['street']) for a in result.history], [(0, 0), (1, 0), (1, 1), (0, 1), (1, 2), (0, 2), (1, 3), (0, 3)])
        first = agents[0].seen[0]
        self.assertEqual(first['to_call'], 10)
        self.assertEqual(first['pot'], 30)
        self.assertEqual(first['players'][0]['stack'], 990)
        self.assertEqual(first['min_raise_to'], 40)

    def test_multiway_order_big_blind_option(self):
        agents = [Scripted() for _ in range(4)]
        result = play_hand(agents, [1000] * 4, button=0)
        self.assertEqual([a['seat'] for a in result.history if a['street'] == 0], [3, 0, 1, 2])
        self.assertEqual([a['seat'] for a in result.history if a['street'] == 1], [1, 2, 3, 0])
        self.assertEqual(agents[2].seen[0]['to_call'], 0)
        self.assertTrue(agents[2].seen[0]['can_raise'])

    def test_side_pots_and_uncalled_refund(self):
        agents = [Scripted({(0, 0): shove}) for _ in range(3)]
        deck = deck_with([cards('Ac Ad'), cards('Kc Kd'), cards('Qc Qd')], cards('2c 4h 6s 8d Tc'))
        result = play_hand(agents, [100, 200, 300], deck=deck)
        self.assertEqual(result.contributions, [100, 200, 200])
        self.assertEqual(result.awards, [300, 200, 0])
        self.assertEqual(result.refunds, [0, 0, 0])  # seat2 calls; no uncalled raise permitted
        self.assertEqual(result.stacks, [300, 200, 100])
        self.assertFalse(result.violations)

    def test_unmatched_raise_refunded(self):
        agents = [Scripted({(0, 0): shove}), Scripted()]
        result = play_hand(agents, [1000, 100], button=0)
        self.assertEqual(result.refunds, [900, 0])
        self.assertEqual(result.contributions, [100, 100])
        self.assertEqual(sum(result.stacks), 1100)

    def test_tied_board_and_odd_chip(self):
        agents = [Scripted({(0, 0): raise_to(4)}), Scripted({(0, 0): {'action': 'fold'}}), Scripted()]
        deck = deck_with([cards('2c 3d'), cards('4c 5d'), cards('6c 7d')], cards('Ts Js Qs Ks As'))
        result = play_hand(agents, [10, 10, 10], small_blind=1, big_blind=2, deck=deck)
        self.assertEqual(result.contributions, [4, 1, 4])
        self.assertEqual(result.awards, [4, 0, 5])
        self.assertEqual(result.stacks, [10, 9, 11])

    def test_short_all_in_does_not_reopen(self):
        agents = [Scripted(), Scripted({(1, 0): raise_to(100)}), Scripted({(1, 0): shove})]
        result = play_hand(agents, [1000, 1000, 250], big_blind=100, small_blind=50)
        second = [o for o in agents[1].seen if o['street'] == 1][1]
        self.assertEqual(second['to_call'], 50)
        self.assertFalse(second['can_raise'])
        self.assertEqual(second['min_raise_to'], 250)
        self.assertFalse(result.violations)

    def test_cumulative_short_all_ins_reopen_per_player(self):
        agents = [Scripted({(1, 0): raise_to(100)}), Scripted({(1, 0): shove}), Scripted(), Scripted({(1, 0): shove}), Scripted()]
        result = play_hand(agents, [1000, 225, 1000, 300, 1000], button=4, big_blind=100, small_blind=50)
        a = [o for o in agents[0].seen if o['street'] == 1][1]
        c = [o for o in agents[2].seen if o['street'] == 1][1]
        self.assertTrue(a['can_raise'])
        self.assertEqual(a['to_call'], 100)
        self.assertEqual(a['min_raise_to'], 300)
        self.assertFalse(c['can_raise'])
        self.assertEqual(c['to_call'], 75)
        self.assertFalse(result.violations)

    def test_short_opening_all_in_checker_not_reopened(self):
        agents = [Scripted(), Scripted({(1, 0): shove}), Scripted()]
        result = play_hand(agents, [1000, 150, 1000], button=2, big_blind=100, small_blind=50)
        unacted = [o for o in agents[2].seen if o['street'] == 1][0]
        checker = [o for o in agents[0].seen if o['street'] == 1][1]
        self.assertTrue(unacted['can_raise'])
        self.assertEqual(unacted['min_raise_to'], 150)
        self.assertFalse(checker['can_raise'])
        self.assertFalse(result.violations)

    def test_short_big_blind_full_bring_in_multiway(self):
        agents = [Scripted() for _ in range(3)]
        result = play_hand(agents, [1000, 1000, 10], button=0, big_blind=20, small_blind=10)
        self.assertEqual(agents[0].seen[0]['to_call'], 20)
        self.assertEqual(agents[1].seen[0]['to_call'], 10)
        self.assertEqual(result.contributions, [20, 20, 10])

    def test_short_big_blind_heads_up_no_extra_wager(self):
        agents = [Scripted(), Scripted()]
        result = play_hand(agents, [1000, 5], button=0, big_blind=20, small_blind=10)
        self.assertEqual(result.history, [])
        self.assertEqual(result.refunds, [5, 0])
        self.assertEqual(result.contributions, [5, 5])
        self.assertEqual(len(result.board), 5)

    def test_illegal_raise_check_and_exception_fallbacks(self):
        class Broken:
            def act(self, obs):
                raise RuntimeError('test')
        for agent in (Scripted({(0, 0): raise_to(30)}), Scripted({(0, 0): {'action': 'check'}}), Broken()):
            result = play_hand([agent, Scripted()], [1000, 1000])
            self.assertEqual(result.history[0]['action'], 'fold')
            self.assertEqual(len(result.violations), 1)
            self.assertEqual(sum(result.stacks), 2000)

    def test_measured_timeout_uses_check_fold_fallback(self):
        agents = [Scripted(), Scripted()]
        result = play_hand(agents, [1000, 1000], enforce_time_limit=True, decision_limit_ns=0)
        self.assertEqual(result.history[0]['action'], 'fold')
        self.assertEqual(result.violations[0]['error'], 'decision timeout')
        self.assertEqual(result.overruns_by_seat, [1, 0])
        self.assertEqual(sum(result.stacks), 2000)

    def test_observation_no_hidden_cards_or_mutation_leak(self):
        class Mutator:
            def act(self, obs):
                self.assertions(obs)
                obs['players'][1]['stack'] = 10**9
                obs['board'].append(51)
                obs['hole'][0] = 51
                obs['history'].append({'action': 'bogus'})
                return {'action': 'call' if obs['to_call'] else 'check'}
            def assertions(self, obs):
                assert 'deck' not in obs and 'holes' not in obs
                assert all('hole' not in p for p in obs['players'])
                assert len(obs['board']) in (0, 3, 4, 5)
        result = play_hand([Mutator(), Scripted()], [1000, 1000])
        self.assertEqual(sum(result.stacks), 2000)
        self.assertEqual(len(result.board), 5)
        self.assertFalse(result.violations)
        self.assertFalse(any(a['action'] == 'bogus' for a in result.history))

    def test_random_legal_games_conserve_chips(self):
        class RandomLegal:
            def __init__(self, seed):
                self.rng = random.Random(seed)
            def act(self, o):
                u = self.rng.random()
                if o['can_raise'] and u < .25:
                    lo, hi = min(o['min_raise_to'], o['max_raise_to']), o['max_raise_to']
                    return raise_to(self.rng.randint(lo, hi))
                if o['to_call'] and u < .45:
                    return {'action': 'fold'}
                return {'action': 'call' if o['to_call'] else 'check'}
        rng = random.Random(16)
        for hand in range(200):
            n = rng.randint(2, 8)
            stacks = [rng.randint(1, 1000) for _ in range(n)]
            result = play_hand([RandomLegal(hand * 11 + i) for i in range(n)], stacks, button=hand % n, seed=hand)
            self.assertEqual(sum(result.stacks), sum(stacks))
            self.assertFalse(result.violations)

    def test_table_scores_and_reset(self):
        class Jam:
            def reset(self, seed=None):
                self.seed = seed
            def act(self, obs):
                return shove(obs)
        agents = [Jam() for _ in range(8)]
        payouts = [50, 30, 20, 0, 0, 0, 0, 0]  # explicitly synthetic
        result = play_table(agents, starting_stack=200, max_hands=200, payouts=payouts, seed=77)
        self.assertEqual(sum(result.stacks), 1600)
        self.assertAlmostEqual(sum(result.points), 100)
        self.assertFalse(result.violations)
        self.assertTrue(all(hasattr(a, 'seed') for a in agents))
        self.assertLess(result.hands, 200)
        self.assertEqual(sorted(result.places)[0], 1)

    def test_headsup_transition_does_not_repeat_big_blind(self):
        from unittest.mock import patch
        from simulator import HandResult
        buttons = []
        def fake_hand(agents, stacks, **kwargs):
            buttons.append(kwargs['button'])
            # On first 3-handed hand button0 busts; previous BB was seat2.
            new_stacks = [0, 150, 150] if len(buttons) == 1 else list(stacks)
            return HandResult(new_stacks, [], [], [[], [], []], [0, 0, 0], [0, 0, 0], [0, 0, 0], [], [], [0, 0, 0], [0, 0, 0])
        with patch('simulator.play_hand', side_effect=fake_hand):
            play_table([Scripted() for _ in range(3)], starting_stack=100, max_hands=3, button=0)
        self.assertEqual(buttons, [0, 2, 1])

    def test_candidate_icm_matches_independent_dp(self):
        from bot.icm import equity
        rng = random.Random(160810)
        for _ in range(100):
            n = rng.randrange(2, 9)
            stacks = [rng.randrange(1, 10000) for i in range(n)]
            prizes = sorted([rng.random() * 100 for i in range(n)], reverse=True)
            values = icm_values(stacks, prizes)
            for i in range(n):
                self.assertAlmostEqual(equity(stacks, prizes, i), values[i], places=9)

    def test_icm_known_cases_and_conservation(self):
        self.assertEqual(icm_values([100, 100], [1, 0]), [.5, .5])
        self.assertEqual(icm_values([300, 100], [1, 0]), [.75, .25])
        values = icm_values([100, 100, 100], [50, 30, 20])
        for value in values:
            self.assertAlmostEqual(value, 100 / 3)
        self.assertAlmostEqual(sum(icm_values([20, 500, 15, 100, 400, 250, 19, 2000], [50, 30, 20, 0, 0, 0, 0, 0])), 100)


class EvaluatorTests(unittest.TestCase):
    def test_named_hands(self):
        cases = [('As Ks Qs Js Ts 2c 3c', (8, 14)),
                 ('Ac Ad Ah As Kc Qc Jc', (7, 14, 13)),
                 ('Ac Ad Ah Kc Kd Kh 2c', (6, 14, 13)),
                 ('Ac 9c 7c 4c 2c Ah Ad', (5, 14, 9, 7, 4, 2)),
                 ('Ac 2d 3h 4s 5c 9d Td', (4, 5)),
                 ('Ac Ad Kh Kd Qc Qd 2c', (2, 14, 13, 12))]
        for text, expected in cases:
            self.assertEqual(evaluate(cards(text)), expected)
            self.assertEqual(evaluate_reference(cards(text)), expected)

    def test_independent_random_5_6_7(self):
        rng = random.Random(160019)
        try:
            from bot.evaluator import evaluate as bot_evaluate
        except ImportError:
            bot_evaluate = None
        for n in (5, 6, 7):
            for _ in range(2000):
                hand = rng.sample(range(52), n)
                rank = evaluate_reference(hand)
                self.assertEqual(evaluate(hand), rank)
                if bot_evaluate:
                    score = rank[0] << 20
                    for r, shift in zip(rank[1:], (16, 12, 8, 4, 0)):
                        score |= r << shift
                    self.assertEqual(bot_evaluate(hand), score, hand)


if __name__ == '__main__':
    unittest.main()
