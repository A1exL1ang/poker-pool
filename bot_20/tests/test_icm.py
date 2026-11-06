import itertools
import math
import random
import unittest

from pokerbot.icm import icm_equities, icm_hero


def permutation_reference(stacks, payouts):
    result = [0.0] * len(stacks)
    active = [i for i, value in enumerate(stacks) if value > 0]
    zero = [i for i, value in enumerate(stacks) if value == 0]
    if zero:
        for i in zero:
            result[i] = sum(payouts[len(active):]) / len(zero)
    for order in itertools.permutations(active):
        remaining = sum(stacks)
        probability = 1.0
        for i in order:
            probability *= stacks[i] / remaining
            remaining -= stacks[i]
        for rank, i in enumerate(order):
            if rank < len(payouts):
                result[i] += probability * payouts[rank]
    return result


class ICMTests(unittest.TestCase):
    def test_equal_stacks_equal_equities_and_conservation(self):
        for value in icm_equities([100] * 4, [50, 30, 20]):
            self.assertAlmostEqual(value, 25.0)

    def test_heads_up_and_winner_take_all(self):
        self.assertEqual(icm_equities([75, 25], [100]), [75.0, 25.0])
        self.assertEqual(icm_equities([75, 25], [70, 30]), [60.0, 40.0])

    def test_permutation_reference(self):
        rng = random.Random(18706)
        for n in range(2, 7):
            for _ in range(4):
                stacks = [rng.randrange(1, 10000) for _ in range(n)]
                payouts = sorted([rng.random() * 50 for _ in range(n)], reverse=True)
                got = icm_equities(stacks, payouts)
                expected = permutation_reference(stacks, payouts)
                for a, b in zip(got, expected):
                    self.assertAlmostEqual(a, b, places=10)
                for hero, value in enumerate(got):
                    self.assertAlmostEqual(icm_hero(stacks, payouts, hero), value, places=10)
                self.assertAlmostEqual(sum(got), sum(payouts), places=10)

    def test_scale_and_permutation_invariance(self):
        stacks = [11, 21, 18, 50]
        payouts = [50, 25, 15, 10]
        expected = icm_equities(stacks, payouts)
        self.assertEqual(icm_equities([x * 7 for x in stacks], payouts), expected)
        perm = [2, 0, 3, 1]
        actual = icm_equities([stacks[i] for i in perm], payouts)
        for a, i in zip(actual, perm):
            self.assertAlmostEqual(a, expected[i])

    def test_zero_stacks_share_residual(self):
        self.assertEqual(icm_equities([100, 0, 0], [60, 30, 10]), [60.0, 20.0, 20.0])
        self.assertEqual(icm_equities([0, 0], [60, 40]), [50.0, 50.0])
        self.assertEqual(icm_equities([], []), [])
        self.assertEqual(icm_hero([100, 0, 0], [60, 30, 10], 1), 20.0)

    def test_one_seat_and_hero(self):
        self.assertEqual(icm_equities([7], [100]), [100.0])
        self.assertEqual(icm_hero([75, 25], [70, 30], 1), 40.0)

    def test_rejects_invalid_numbers_and_size(self):
        for stacks, payouts in [([-1, 2], []), ([math.nan], []), ([1], [math.inf]),
                                ([1], [1, 2]), ([1] * 9, [])]:
            with self.assertRaises(ValueError):
                icm_equities(stacks, payouts)


class PolicyICMThresholdTests(unittest.TestCase):
    """Terminal calls independently checked against enumerated finish orders.

    Payouts below are hypothetical unit tests, never official contest awards.
    """

    @staticmethod
    def obs(hero_behind=4000, hero_committed=0, villain_committed=4000,
            third_behind=4000, third_committed=0, payouts=None):
        return {"hero": 0, "stacks": [hero_behind, 0, third_behind],
                "committed": [hero_committed, villain_committed, third_committed],
                "pot": hero_committed + villain_committed + third_committed,
                "hand_start_stacks": [hero_behind + hero_committed,
                                      villain_committed, third_behind + third_committed],
                "payouts": payouts or [.5, .3, .2]}

    def test_equal_stack_bubble_call_needs_four_sevenths(self):
        from pokerbot.policy import Agent
        agent = Agent()
        obs = self.obs()
        threshold = agent._call_threshold(obs, 4000, [1], terminal=True)
        self.assertAlmostEqual(threshold, 4 / 7, places=12)

    def test_sidepot_refund_and_dead_money_terminal_threshold(self):
        from pokerbot.policy import Agent
        agent = Agent()
        # Hero has 1500 behind after contributing 500, villain has committed
        # 4000, and a folded third seat has 3000 dead chips plus 1000 behind.
        # Winning awards hero 6000 and refunds/awards villain the excess 3000.
        obs = self.obs(1500, 500, 4000, 1000, 3000)
        fold = permutation_reference([1500, 7500, 1000], [.5, .3, .2])[0]
        lose = permutation_reference([0, 9000, 1000], [.5, .3, .2])[0]
        win = permutation_reference([6000, 3000, 1000], [.5, .3, .2])[0]
        expected = (fold - lose) / (win - lose)
        self.assertAlmostEqual(agent._call_threshold(obs, 1500, [1], True), expected, places=12)

    def test_winner_take_all_reduces_to_chip_odds(self):
        from pokerbot.policy import Agent
        agent = Agent()
        obs = self.obs(1500, 500, 4000, 1000, 3000, [1., 0., 0.])
        self.assertAlmostEqual(agent._call_threshold(obs, 1500, [1], True), 1500 / 6000)

    def test_missing_payouts_uses_chip_odds(self):
        from pokerbot.policy import Agent
        agent = Agent()
        obs = self.obs(1500, 500, 4000, 1000, 3000)
        obs["payouts"] = None
        self.assertEqual(agent._call_threshold(obs, 1500, [1], True), 1500 / 6000)


if __name__ == "__main__":
    unittest.main()
