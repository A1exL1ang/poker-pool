import math
import random
import time
import unittest

from engine import Hand, evaluate_reference, rank_five, settle_pots, icm, tournament


def cards(text):
    return [('23456789TJQKA'.index(x[0]) * 4 + 'cdhs'.index(x[1])) for x in text.split()]


class Passive:
    def act(self, obs):
        return {'action': 'call' if obs['to_call'] else 'check'}


class RandomLegal:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, obs):
        x = self.rng.random()
        if obs['can_raise'] and x < 0.25:
            minimum, maximum = obs['min_raise_to'], obs['max_raise_to']
            return {'action': 'raise', 'amount': self.rng.randint(min(minimum, maximum), maximum)}
        if obs['to_call'] and x < 0.45:
            return {'action': 'fold'}
        return {'action': 'call' if obs['to_call'] else 'check'}


class EngineTests(unittest.TestCase):
    def test_reference_categories_and_wheel(self):
        hands = ['As Ks Qs Js Ts', 'Ac Ad Ah As 2c', 'Kc Kd Kh 2s 2d',
                 'Ac Jc 9c 7c 3c', '9c 8d 7h 6s 5c', 'Qc Qd Qh 9s 2d',
                 'Ac Ad Kh Ks 2d', 'Ac Ad Qh Js 2d', 'Ac Kh Qs Jd 9c']
        ranks = [rank_five(cards(h)) for h in hands]
        self.assertEqual(ranks, sorted(ranks, reverse=True))
        self.assertEqual(rank_five(cards('Ac 2d 3h 4s 5c')), (4, 5))
        self.assertGreater(rank_five(cards('2c 3d 4h 5s 6c')), rank_five(cards('Ac 2d 3h 4s 5c')))
        self.assertEqual(evaluate_reference(cards('Ac Ad Ah As Kc Kd Kh')), (7, 14, 13))
        self.assertEqual(evaluate_reference(cards('Ac Ad Ah Kc Kd Kh 2s')), (6, 14, 13))

    def test_deal_and_observation_privacy(self):
        h = Hand([2000] * 8, 0, 10, 20, rng=random.Random(1))
        self.assertEqual((h.sb_seat, h.bb_seat, h.actor), (1, 2, 3))
        obs = h.observation()
        self.assertEqual(obs['pot'], 30)
        self.assertEqual(obs['to_call'], 20)
        self.assertEqual(obs['min_raise_to'], 40)
        self.assertEqual(obs['max_raise_to'], 2000)
        self.assertNotIn('holes', obs)
        obs['hole'][0] = -1
        obs['stacks'][0] = -1
        self.assertNotIn(-1, h.holes[h.actor])
        self.assertNotIn(-1, h.stacks)

    def test_heads_up_preflop_button_first_postflop_bb_first(self):
        h = Hand([100, 100], 0, 5, 10)
        self.assertEqual((h.sb_seat, h.bb_seat, h.actor), (0, 1, 0))
        h.step({'action': 'call'})
        self.assertEqual(h.actor, 1)
        h.step({'action': 'check'})
        self.assertEqual((h.street, h.actor, len(h.board)), (1, 1, 3))
        h.step({'action': 'check'})
        self.assertEqual(h.actor, 0)

    def test_big_blind_gets_option_after_all_limp(self):
        h = Hand([100] * 3, 0, 5, 10)
        h.step({'action': 'call'})
        h.step({'action': 'call'})
        self.assertEqual(h.actor, 2)
        self.assertEqual(h.street, 0)
        self.assertEqual(h.observation()['to_call'], 0)
        h.step({'action': 'raise', 'amount': 20})
        self.assertEqual(h.actor, 0)

    def test_fold_ends_without_private_showdown(self):
        h = Hand([100, 100], 0, 5, 10)
        h.step({'action': 'fold'})
        self.assertTrue(h.done)
        self.assertEqual(h.board, [])
        self.assertEqual(h.stacks, [95, 105])

    def test_minimum_raise_and_atomic_invalid(self):
        h = Hand([100] * 3, 0, 5, 10)
        before = h.observation()
        for action in [{'action': 'check'}, {'action': 'raise', 'amount': 19},
                       {'action': 'raise', 'amount': 101}, {'action': 'raise', 'amount': 20.0}]:
            with self.assertRaises(ValueError):
                h.step(action)
            self.assertEqual(h.observation(), before)
        h.step({'action': 'raise', 'amount': 25})
        self.assertEqual(h.observation()['min_raise_to'], 40)

    def test_short_all_in_does_not_reopen(self):
        h = Hand([200, 200, 35], 0, 5, 10)
        h.step({'action': 'raise', 'amount': 30})  # 20 full increment.
        h.step({'action': 'call'})
        h.step({'action': 'raise', 'amount': 35})
        self.assertEqual(h.actor, 0)
        self.assertFalse(h.observation()['can_raise'])
        self.assertEqual(h.observation()['min_raise_to'], 55)
        with self.assertRaises(ValueError):
            h.step({'action': 'raise', 'amount': 55})
        h.step({'action': 'call'})
        self.assertFalse(h.observation()['can_raise'])
        h.step({'action': 'call'})
        self.assertEqual(h.street, 1)

    def test_multiple_short_allins_cumulatively_reopen(self):
        h = Hand([150, 200, 200, 140], 0, 5, 10)
        self.assertEqual(h.actor, 3)
        h.step({'action': 'raise', 'amount': 100})
        h.step({'action': 'raise', 'amount': 150})  # Short, full increment remains 90.
        h.step({'action': 'call'})
        h.step({'action': 'raise', 'amount': 200})  # Another short, cumulative 100.
        self.assertEqual(h.actor, 3)
        self.assertFalse(h.observation()['can_raise'])  # His 40 behind cannot exceed 200.
        # Repeat with original bettor deep enough to prove reopening.
        h = Hand([150, 500, 200, 500], 0, 5, 10)
        h.step({'action': 'raise', 'amount': 100})
        h.step({'action': 'raise', 'amount': 150})
        h.step({'action': 'call'})
        h.step({'action': 'raise', 'amount': 200})
        self.assertTrue(h.observation()['can_raise'])
        self.assertEqual(h.observation()['min_raise_to'], 290)
        h.step({'action': 'raise', 'amount': 290})

    def test_player_not_yet_acted_can_raise_over_short_allin(self):
        h = Hand([200, 35, 200, 200], 0, 5, 10)
        h.step({'action': 'raise', 'amount': 30})  # Seat3
        h.step({'action': 'call'})  # Seat0
        h.step({'action': 'raise', 'amount': 35})  # Seat1 short
        self.assertEqual(h.actor, 2)
        self.assertTrue(h.observation()['can_raise'])
        h.step({'action': 'raise', 'amount': 55})
        self.assertTrue(h.observation()['can_raise'])

    def test_under_blind_allins_no_unmatched_betting(self):
        h = Hand([100, 3], 0, 5, 10)
        self.assertTrue(h.done)
        self.assertEqual(len(h.board), 5)
        self.assertEqual(sum(h.stacks), 103)
        h = Hand([100, 8], 0, 5, 10)
        self.assertFalse(h.done)
        self.assertEqual(h.observation()['to_call'], 3)
        self.assertFalse(h.observation()['can_raise'])
        h.step({'action': 'call'})
        self.assertTrue(h.done)
        self.assertEqual(h.contributions, [8, 8])

    def test_sidepots_uncalled_folded_dead_money_and_odd_chips(self):
        # Seat0 wins main 120; seat1 wins next 80, and gets own uncalled50.
        self.assertEqual(settle_pots([40, 130, 80], [False]*3, [3, 2, 1]), [120, 130, 0])
        self.assertEqual(settle_pots([10, 10, 5], [False, False, True], [2, 2, None], button=0), [12, 13, 0])
        self.assertEqual(settle_pots([10, 10, 5], [False, False, True], [2, 2, None], button=1), [13, 12, 0])
        self.assertEqual(settle_pots([100, 100, 30], [True, False, False], [None, 1, 2]), [0, 140, 90])

    def test_no_raise_when_only_other_players_allin(self):
        h = Hand([100, 60], 0, 5, 10)
        h.step({'action': 'call'})
        h.step({'action': 'raise', 'amount': 60})
        self.assertFalse(h.observation()['can_raise'])
        h.step({'action': 'call'})
        self.assertTrue(h.done)
        self.assertEqual(len(h.board), 5)

    def test_no_raise_when_remaining_stack_cannot_contest_increase(self):
        h = Hand([100, 50, 200], 0, 5, 10)
        h.step({'action': 'raise', 'amount': 100})
        h.step({'action': 'call'})
        self.assertEqual(h.actor, 2)
        self.assertFalse(h.observation()['can_raise'])
        with self.assertRaises(ValueError):
            h.step({'action': 'raise', 'amount': 200})
        h.step({'action': 'call'})
        self.assertTrue(h.done)

    def test_full_checkdown_has_four_streets(self):
        h = Hand([100] * 3, 0, 5, 10)
        h.play([Passive()] * 3)
        self.assertEqual(len(h.board), 5)
        self.assertEqual(len(h.history), 12)
        self.assertEqual(h.errors, 0)
        self.assertEqual(sum(h.stacks), 300)

    def test_invalid_agent_and_exception_safe_fallback(self):
        class Broken:
            def act(self, obs):
                raise RuntimeError('test')
        h = Hand([100] * 3, 0, 5, 10)
        h.play([Broken()] * 3)
        self.assertEqual(h.errors, 2)
        self.assertEqual(sum(h.stacks), 300)

    def test_timeout_and_illegal_action_fallbacks_are_counted(self):
        class Slow:
            def act(self, obs):
                time.sleep(.002)
                return {'action': 'call'}
        h = Hand([100] * 3, 0, 5, 10)
        h.play([Slow()] * 3, enforce_time_limit=True, record_latencies=True)
        self.assertEqual(h.timeouts, 2)
        self.assertEqual(h.errors, 0)
        self.assertEqual(sum(h.timeouts_by_seat), 2)
        self.assertEqual(sum(h.decisions_by_seat), 2)
        class Illegal:
            def act(self, obs):
                return {'action': 'check'}
        h = Hand([100] * 3, 0, 5, 10)
        h.play([Illegal()] * 3)
        self.assertEqual(h.errors, 2)
        self.assertEqual(sum(h.errors_by_seat), 2)

    def test_exact_icm(self):
        self.assertEqual(icm([100, 100], [1, 0]), [0.5, 0.5])
        self.assertEqual(icm([100, 100, 100], [9, 6, 0]), [5, 5, 5])
        self.assertEqual(icm([100, 0, 300], [1, 0]), [0.25, 0.0, 0.75])
        for n in range(2, 9):
            v = icm(list(range(1, n+1)), list(range(n, 0, -1)))
            self.assertAlmostEqual(sum(v), n*(n+1)/2)
            self.assertEqual(v, sorted(v))

    def test_random_hands_conserve_chips_and_all_actions_legal(self):
        rng = random.Random(98)
        for trial in range(150):
            n = rng.randint(2, 8)
            stacks = [rng.randint(1, 500) for _ in range(n)]
            h = Hand(stacks, rng.randrange(n), 5, 10, rng=rng)
            h.play([RandomLegal(trial*10+i) for i in range(n)])
            self.assertEqual(sum(h.stacks), sum(stacks))
            self.assertTrue(all(isinstance(s, int) and s >= 0 for s in h.stacks))
            self.assertEqual(h.errors, 0)
            all_cards = h.board + [c for hole in h.holes for c in hole]
            self.assertEqual(len(set(all_cards)), len(all_cards))

    def test_tournament_terminal_points_conservation(self):
        result = tournament([Passive()] * 8, seed=5, payouts=[8, 7, 6, 5, 4, 3, 2, 1],
                            blind_schedule=[(5, 10), (10, 20)], max_hands=30, initial_stack=100)
        self.assertEqual(sum(result['stacks']), 800)
        self.assertAlmostEqual(sum(result['points']), 36)
        self.assertEqual(result['errors'], 0)
        self.assertLessEqual(result['hands'], 30)
        for distribution in result['finish_probabilities']:
            self.assertAlmostEqual(sum(distribution), 1.)
        for place in range(8):
            self.assertAlmostEqual(sum(p[place] for p in result['finish_probabilities']), 1.)
        result = tournament([Passive()] * 2, seed=1, payouts=[1, 0],
                            blind_schedule=[(100, 200)], initial_stack=10)
        self.assertEqual(result['hands'], 1)
        self.assertEqual(sum(result['points']), 1)

    def test_simultaneous_equal_stack_busts_split_places(self):
        result = tournament([Passive()] * 8, seed=42, payouts=[8, 7, 6, 5, 4, 3, 2, 1],
                            blind_schedule=[(100, 200)], max_hands=1, initial_stack=10,
                            evaluator=lambda cs: cs[0])
        self.assertEqual(sorted(result['stacks']), [0]*7 + [80])
        self.assertEqual(sorted(result['points']), [4]*7 + [8])

    def test_terminal_board_tie_and_unmatched_refund(self):
        h = Hand([100, 50, 20], 0, 5, 10, evaluator=lambda cs: 1)
        h.step({'action': 'raise', 'amount': 100})
        h.step({'action': 'call'})
        h.step({'action': 'call'})
        self.assertTrue(h.done)
        self.assertEqual(h.stacks, [100, 50, 20])


if __name__ == '__main__':
    unittest.main()
