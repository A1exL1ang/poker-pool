import itertools
import random
import unittest
from unittest.mock import patch

from benchmarks.engine import Hand, evaluate, icm, next_hand_positions, play_table
from benchmarks.opponents import BaselineAgent


def cards(text):
    return [('23456789TJQKA'.index(x[0]) * 4 + 'cdhs'.index(x[1])) for x in text.split()]


def reference5(cs):
    counts = {}
    for c in cs:
        r = c // 4 + 2
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    descending = sorted(counts, reverse=True)
    is_flush = len({c % 4 for c in cs}) == 1
    straight_high = (5 if descending == [14, 5, 4, 3, 2] else
                     descending[0] if len(descending) == 5 and descending[0] - descending[-1] == 4 else 0)
    if is_flush and straight_high:
        return (8, straight_high)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if [g[0] for g in groups] == [3, 2]:
        return (6, groups[0][1], groups[1][1])
    if is_flush:
        return (5, *descending)
    if straight_high:
        return (4, straight_high)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for n, r in groups[1:]), reverse=True))
    if [g[0] for g in groups[:2]] == [2, 2]:
        return (2, *sorted((r for n, r in groups[:2]), reverse=True), groups[2][1])
    if groups[0][0] == 2:
        return (1, groups[0][1], *sorted((r for n, r in groups[1:]), reverse=True))
    return (0, *descending)


class Calling:
    def __init__(self):
        self.observations = []

    def act(self, obs):
        self.observations.append(obs)
        return {'action': 'call' if obs['to_call'] else 'check',
                'amount': min(obs['to_call'], obs['stacks'][obs['hero']])}


def fixture(stacks=(1000, 1000, 1000, 1000)):
    hand = Hand(list(stacks), [Calling() for _ in stacks], 0, 10, 20, 4)
    hand.preflop_order = list(range(len(stacks)))
    return hand


class EngineTests(unittest.TestCase):
    def test_evaluator_known_categories(self):
        examples = [('As Ks Qs Js Ts 2c 3d', 8), ('Ac Ad Ah As Kc 2c 3d', 7),
                    ('Ac Ad Ah Kc Kd 2c 3d', 6), ('Ac Jc 9c 5c 3c Kd Qh', 5),
                    ('Ac 2d 3h 4s 5c Kd Qh', 4), ('Ac Ad Ah Kc Qd 2c 3d', 3),
                    ('Ac Ad Kh Ks Qc 2c 3d', 2), ('Ac Ad Kh Qs Jc 2c 3d', 1),
                    ('Ac Kd Jh 9s 7c 2c 3d', 0)]
        for text, category in examples:
            self.assertEqual(evaluate(cards(text))[0], category)

    def test_evaluator_matches_independent_five_card_enumeration(self):
        rng = random.Random(18876)
        for _ in range(1500):
            hand = rng.sample(range(52), rng.choice([5, 6, 7]))
            self.assertEqual(evaluate(hand), max(reference5(cs) for cs in itertools.combinations(hand, 5)))

    def test_double_trips_full_house(self):
        self.assertEqual(evaluate(cards('Ac Ad Ah Kc Kd Kh 2c')), (6, 14, 13))

    def test_flush_not_confused_by_offsuit_straight(self):
        self.assertEqual(evaluate(cards('Ac Jc 9c 5c 3c 4d 2h'))[0], 5)

    def test_raise_amount_is_total(self):
        h = fixture()
        h.put(0, 10)
        h.put(1, 20)
        h.apply(0, {'action': 'raise', 'amount': 60})
        self.assertEqual(h.contributions[0], 60)
        self.assertEqual(h.stacks[0], 940)
        self.assertEqual(h.min_raise_to(), 100)

    def test_full_raise_reopens_action(self):
        h = fixture()
        h.apply(0, {'action': 'call', 'amount': 20})
        h.apply(1, {'action': 'raise', 'amount': 60})
        self.assertTrue(h.can_raise(0))

    def test_single_short_raise_does_not_reopen(self):
        h = fixture((1000, 30, 1000, 1000))
        h.apply(0, {'action': 'call', 'amount': 20})
        h.apply(1, {'action': 'raise', 'amount': 30})
        self.assertFalse(h.can_raise(0))
        self.assertTrue(h.can_raise(2))
        self.assertEqual(h.min_raise_to(), 50)

    def test_cumulative_short_raises_reopen(self):
        h = fixture((1000, 30, 40, 1000))
        h.apply(0, {'action': 'call', 'amount': 20})
        h.apply(1, {'action': 'raise', 'amount': 30})
        h.apply(2, {'action': 'raise', 'amount': 40})
        self.assertTrue(h.can_raise(0))
        self.assertEqual(h.min_raise_to(), 60)

    def test_recent_caller_not_reopened_by_same_short_raises(self):
        h = fixture((1000, 30, 40, 1000))
        h.apply(0, {'action': 'call', 'amount': 20})
        h.apply(1, {'action': 'raise', 'amount': 30})
        h.apply(3, {'action': 'call', 'amount': 30})
        h.apply(2, {'action': 'raise', 'amount': 40})
        self.assertTrue(h.can_raise(0))
        self.assertFalse(h.can_raise(3))

    def test_short_non_allin_raise_rejected(self):
        h = fixture()
        h.apply(0, {'action': 'raise', 'amount': 30})
        self.assertEqual(h.invalid_actions, 1)
        self.assertFalse(h.in_hand[0])

    def test_short_opening_allin_requires_full_raise_and_does_not_reopen_checker(self):
        h = fixture((1000, 10, 1000, 1000))
        h.high = 0
        h.street = 1
        h.apply(0, {'action': 'check', 'amount': 0})
        h.apply(1, {'action': 'raise', 'amount': 10})
        self.assertEqual(h.min_raise_to(), 30)
        self.assertFalse(h.can_raise(0))
        self.assertTrue(h.can_raise(2))
        h.apply(2, {'action': 'raise', 'amount': 30})
        self.assertEqual(h.min_raise_to(), 50)
        self.assertTrue(h.can_raise(0))

    def test_no_raise_without_opponent_who_can_respond(self):
        h = fixture((1000, 20))
        h.put(1, 20)
        self.assertFalse(h.can_raise(0))

    def test_no_raise_when_opponent_can_only_call_allin(self):
        h = fixture((1000, 10))
        self.assertFalse(h.can_raise(0))

    def test_call_capped_to_stack(self):
        h = fixture((5, 1000))
        h.apply(0, {'action': 'call', 'amount': 5})
        self.assertEqual(h.stacks[0], 0)
        self.assertEqual(h.contributions[0], 5)

    def test_heads_up_button_small_blind_acts_first_preflop(self):
        agents = [Calling(), Calling()]
        seen = []
        h = Hand([1000, 1000], agents, 0, 10, 20, 5,
                 observer=lambda obs, action, elapsed: seen.append((obs['street'], obs['hero'])))
        h.play()
        self.assertEqual(seen[:2], [(0, 0), (0, 1)])
        self.assertEqual(next(x for x in seen if x[0] == 1), (1, 1))

    def test_eight_handed_action_order(self):
        agents = [Calling() for _ in range(8)]
        seen = []
        h = Hand([1000] * 8, agents, 0, 10, 20, 5,
                 observer=lambda obs, action, elapsed: seen.append((obs['street'], obs['hero'])))
        h.play()
        self.assertEqual([seat for street, seat in seen if street == 0], [3, 4, 5, 6, 7, 0, 1, 2])
        self.assertEqual([seat for street, seat in seen if street == 1], [1, 2, 3, 4, 5, 6, 7, 0])

    def test_short_big_blind_preserves_nominal_bringin(self):
        agents = [Calling(), Calling(), Calling()]
        h = Hand([1000, 1000, 5], agents, 0, 10, 20, 5)
        h.play()
        self.assertEqual(agents[0].observations[0]['to_call'], 20)
        self.assertEqual(agents[0].observations[0]['min_raise_to'], 40)

    def test_side_pots_and_uncalled_excess(self):
        h = fixture((100, 200, 300))
        for i, amount in enumerate([100, 200, 300]):
            h.put(i, amount)
        h.board = cards('2c 3d 4h 8s 9c')
        h.hole = [cards('Ac Ad'), cards('Kc Kd'), cards('Qc Qd')]
        self.assertEqual(h.settle(), [300, 200, 100])
        self.assertEqual([p['amount'] for p in h.pots], [300, 200, 100])

    def test_folded_best_hand_cannot_win(self):
        h = fixture((100, 100, 100))
        for i in range(3):
            h.put(i, 100)
        h.board = cards('2c 3d 4h 8s 9c')
        h.hole = [cards('Ac Ad'), cards('Kc Kd'), cards('Qc Qd')]
        h.in_hand[0] = False
        self.assertEqual(h.settle(), [0, 300, 0])

    def test_board_tie_and_odd_chip_left_of_button(self):
        h = fixture((2, 2, 1))
        h.button = 2
        for i, amount in enumerate([2, 2, 1]):
            h.put(i, amount)
        h.board = cards('Ac Kc Qc Jc Tc')
        h.hole = [cards('2d 3d'), cards('4d 5d'), cards('6d 7d')]
        h.in_hand[2] = False
        self.assertEqual(h.settle(), [3, 2, 0])

    def test_fold_win_needs_no_showdown_cards(self):
        h = fixture((100, 100, 100))
        h.put(0, 30)
        h.put(1, 20)
        h.put(2, 10)
        h.in_hand = [True, False, False]
        self.assertEqual(h.settle(), [130, 80, 90])

    def test_icm_equal_stacks(self):
        values = icm([100] * 8, [50, 30, 20, 0, 0, 0, 0, 0])
        for value in values:
            self.assertAlmostEqual(value, 12.5)

    def test_icm_winner_take_all_equals_chip_fraction(self):
        stacks = [3, 7, 13, 29]
        self.assertEqual(icm(stacks, [52, 0, 0, 0]), [float(x) for x in stacks])

    def test_icm_heads_up_known(self):
        self.assertEqual(icm([30, 70], [100, 20]), [44, 76])

    def test_icm_conserves_points(self):
        self.assertAlmostEqual(sum(icm([1, 2, 4, 8, 16, 32, 64, 128], [8, 7, 6, 5, 4, 3, 2, 1])), 36)

    def test_random_legal_play_conserves_chips(self):
        for seed in range(20):
            agents = [BaselineAgent(style, seed * 8 + i) for i, style in enumerate(
                ['random', 'balanced', 'calling', 'aggressive', 'shove', 'tight', 'random', 'calling'])]
            result = play_table(agents, seed, max_hands=30, payouts=[50, 30, 20, 0, 0, 0, 0, 0])
            self.assertEqual(sum(result['stacks']), 16000)
            self.assertAlmostEqual(sum(result['points']), 100)
            self.assertEqual(result['errors'], 0)
            self.assertEqual(result['invalid_actions'], 0)
            self.assertLessEqual(result['hands'], 30)

    def test_action_observation_has_no_hidden_cards(self):
        h = fixture()
        h.hole = [cards('Ac Ad'), cards('Kc Kd'), cards('Qc Qd'), cards('Jc Jd')]
        obs = h.observation(0)
        self.assertEqual(obs['hole'], h.hole[0])
        self.assertNotIn('deck', obs)
        self.assertNotIn('opponent_hole', obs)

    def test_exception_safe_fallback(self):
        class Broken:
            def act(self, obs):
                raise ValueError('intentional')
        h = Hand([1000, 1000], [Broken(), Calling()], 0, 10, 20, 2)
        h.play()
        self.assertEqual(h.errors, 1)
        self.assertEqual(h.stacks, [990, 1010])

    def test_hand_cap_and_deterministic_decks(self):
        args = dict(seed=9123, max_hands=7, payouts=[50, 30, 20, 0, 0, 0, 0, 0])
        a = play_table([Calling() for _ in range(8)], **args)
        b = play_table([Calling() for _ in range(8)], **args)
        self.assertEqual(a, b)
        self.assertEqual(a['hands'], 7)

    def test_simultaneous_bust_uses_starting_stacks(self):
        calls = [0]

        def scripted(hand):
            hand.bb_seat = 2
            hand.stacks[:] = [1000, 2000, 3000] if calls[0] == 0 else [0, 0, 6000]
            calls[0] += 1
            return hand.stacks

        with patch.object(Hand, 'play', scripted):
            result = play_table([Calling() for _ in range(3)], 12, payouts=[60, 30, 10])
        self.assertEqual(result['points'], [10, 30, 60])
        self.assertEqual(result['finish'], [3, 2, 1])
        self.assertEqual(result['hands'], 2)

    def test_equal_stack_simultaneous_bust_splits_points(self):
        def scripted(hand):
            hand.bb_seat = 2
            hand.stacks[:] = [0, 0, 6000]
            return hand.stacks

        with patch.object(Hand, 'play', scripted):
            result = play_table([Calling() for _ in range(3)], 12, payouts=[60, 30, 10])
        self.assertEqual(result['points'], [20, 20, 60])
        self.assertEqual(result['finish'], [2.5, 2.5, 1])
        self.assertEqual(result['hands'], 1)

    def test_latency_overrun_uses_fold_fallback(self):
        h = Hand([1000, 1000], [Calling(), Calling()], 0, 10, 20, 2, decision_limit_ns=1000000)
        with patch('benchmarks.engine.time.perf_counter_ns', side_effect=[0, 2000000]):
            h.play()
        self.assertEqual(h.overruns, 1)
        self.assertEqual(h.stacks, [990, 1010])

    def test_all_in_runout_no_extra_decisions(self):
        h = Hand([10, 20], [Calling(), Calling()], 0, 10, 20, 2)
        h.play()
        self.assertEqual(h.decisions, 0)
        self.assertEqual(len(h.board), 5)
        self.assertEqual(sum(h.stacks), 30)

    def test_runtime_evaluator_agrees_with_independent_engine(self):
        from bot.evaluator import evaluate as runtime_evaluate
        rng = random.Random(771499)
        for _ in range(5000):
            cs = rng.sample(range(52), rng.choice([5, 6, 7]))
            rank = evaluate(cs)
            expected = rank[0] * 15 ** 5
            expected += sum(value * 15 ** (4 - i) for i, value in enumerate(rank[1:]))
            self.assertEqual(runtime_evaluate(cs), expected)

    def test_busted_big_blind_leaves_dead_small_blind(self):
        self.assertEqual(next_hand_positions([100, 100, 0, 100], 2), (1, None, 3))
        seen = []
        h = Hand([100, 100, 0, 100], [Calling() for _ in range(4)], 1, 10, 20, 5,
                 blind_seats=(None, 3),
                 observer=lambda obs, action, elapsed: seen.append(obs))
        h.play()
        self.assertEqual(seen[0]['hero'], 0)
        self.assertEqual(seen[0]['pot'], 20)
        self.assertEqual(seen[0]['street_bets'], [0, 0, 0, 20])

    def test_busted_small_blind_leaves_dead_button(self):
        self.assertEqual(next_hand_positions([100, 0, 100, 100], 2), (1, 2, 3))
        seen = []
        h = Hand([100, 0, 100, 100], [Calling() for _ in range(4)], 1, 10, 20, 5,
                 blind_seats=(2, 3), observer=lambda obs, action, elapsed: seen.append(obs))
        h.play()
        self.assertEqual(seen[0]['button'], 1)
        self.assertEqual(seen[0]['hero'], 0)
        self.assertEqual(seen[0]['street_bets'], [0, 0, 10, 20])

    def test_heads_up_transition_preserves_next_big_blind(self):
        self.assertEqual(next_hand_positions([0, 100, 100], 2), (2, 2, 1))
        self.assertEqual(next_hand_positions([100, 0, 100], 2), (2, 2, 0))
        self.assertEqual(next_hand_positions([100, 100, 0], 2), (1, 1, 0))


if __name__ == '__main__':
    unittest.main()
