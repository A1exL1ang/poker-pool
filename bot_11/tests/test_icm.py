import itertools
import random
import unittest

from pokerbot.icm import call_threshold, utility


def brute_utility(stacks, hero, payouts):
    """Independent full finish-permutation reference (at most eight seats)."""
    total = sum(stacks)
    expected = 0.0
    for order in itertools.permutations(range(len(stacks))):
        probability = 1.0
        remaining = total
        for seat in order:
            probability *= stacks[seat] / remaining
            remaining -= stacks[seat]
        expected += probability * payouts[order.index(hero)]
    return expected


def original_dp_utility(stacks, hero, payouts):
    """Original O(n*2**n) subset-order DP, retained as an independent oracle."""
    others = [stack for seat, stack in enumerate(stacks) if seat != hero]
    size = 1 << len(others)
    totals = [0] * size
    probabilities = [0.0] * size
    probabilities[0] = 1.0
    result = 0.0
    for mask in range(1, size):
        bit = mask & -mask
        totals[mask] = totals[mask ^ bit] + others[bit.bit_length() - 1]
    for mask in range(size):
        remaining = sum(stacks) - totals[mask]
        result += probabilities[mask] * stacks[hero] / remaining * payouts[mask.bit_count()]
        for seat, stack in enumerate(others):
            if not mask & (1 << seat):
                probabilities[mask | (1 << seat)] += probabilities[mask] * stack / remaining
    return result


def brute_call_threshold(stacks, hero, villain, pot, cost, payouts):
    live = [seat for seat, stack in enumerate(stacks) if stack > 0 or seat in (hero, villain)]
    h, v = live.index(hero), live.index(villain)
    start = [stacks[seat] for seat in live]
    pay = list(payouts[:len(start)])

    def resolved_value(chips):
        if chips[h] <= 0:
            return pay[-1]
        survivors = [seat for seat, stack in enumerate(chips) if stack > 0]
        return brute_utility([chips[seat] for seat in survivors], survivors.index(h), pay[:len(survivors)])

    folded = start[:]
    folded[v] += pot
    lost = start[:]
    lost[h] -= cost
    lost[v] += pot + cost
    won = start[:]
    won[h] += pot
    uf, ul, uw = map(resolved_value, (folded, lost, won))
    return max(0.0, min(1.0, (uf - ul) / (uw - ul))) if uw > ul else 1.0


class ICMTests(unittest.TestCase):
    def test_subset_formula_against_original_dp(self):
        rng = random.Random(11040)
        for size in range(1, 9):
            for _ in range(80):
                stacks = [rng.randint(1, 1000000) for _ in range(size)]
                payouts = sorted([rng.uniform(-10, 100) for _ in range(size)], reverse=True)
                hero = rng.randrange(size)
                self.assertAlmostEqual(utility(stacks, hero, payouts), original_dp_utility(stacks, hero, payouts), places=10)
        # Sharp payout steps cause alternating finite differences; exercise
        # their cancellation against the positive-probability DP reference.
        for paid in range(1, 9):
            payouts = [100] * paid + [0] * (8 - paid)
            for hero in range(8):
                stacks = [1, 3, 11, 99, 1000, 23, 47, 12000]
                self.assertAlmostEqual(utility(stacks, hero, payouts), original_dp_utility(stacks, hero, payouts), places=10)

    def test_exact_utility_against_full_finish_permutations(self):
        rng = random.Random(11041)
        for size in range(2, 7):
            for _ in range(7):
                stacks = [rng.randint(1, 1000) for _ in range(size)]
                payouts = sorted([rng.randint(0, 100) for _ in range(size)], reverse=True)
                hero = rng.randrange(size)
                self.assertAlmostEqual(utility(stacks, hero, payouts), brute_utility(stacks, hero, payouts), places=10)
        stacks = [32, 97, 3, 48, 150, 11, 69, 14]
        payouts = [100, 65, 40, 20, 10, 5, 2, 0]
        self.assertAlmostEqual(utility(stacks, 5, payouts), brute_utility(stacks, 5, payouts), places=10)

    def test_equal_stacks_conservation_and_scale(self):
        for size in range(2, 9):
            payouts = list(range(size - 1, -1, -1))
            stacks = [100] * size
            for hero in range(size):
                self.assertAlmostEqual(utility(stacks, hero, payouts), sum(payouts) / size)
            stacks = [37 * (i + 1) for i in range(size)]
            values = [utility(stacks, hero, payouts) for hero in range(size)]
            self.assertAlmostEqual(sum(values), sum(payouts), places=10)
            self.assertAlmostEqual(values[0], utility([stack * 100 for stack in stacks], 0, payouts), places=10)

    def test_winner_take_all_is_chip_fraction(self):
        rng = random.Random(11042)
        for size in range(2, 9):
            stacks = [rng.randint(1, 10000) for _ in range(size)]
            payouts = [100] + [0] * (size - 1)
            for hero in range(size):
                self.assertAlmostEqual(utility(stacks, hero, payouts), 100 * stacks[hero] / sum(stacks), places=11)

    def test_call_threshold_matches_brute_outcomes(self):
        rng = random.Random(11043)
        for size in range(2, 7):
            for _ in range(12):
                stacks = [rng.randint(10, 1000) for _ in range(size)]
                hero, villain = rng.sample(range(size), 2)
                pot = rng.randint(1, 1000)
                cost = rng.choice((stacks[hero], rng.randint(1, stacks[hero])))
                payouts = sorted([rng.randint(0, 100) for _ in range(size)], reverse=True)
                expected = brute_call_threshold(stacks, hero, villain, pot, cost, payouts)
                self.assertAlmostEqual(call_threshold(stacks, hero, villain, pot, cost, payouts), expected, places=10)

    def test_winner_take_all_call_threshold_is_pot_odds(self):
        rng = random.Random(11044)
        for size in range(2, 9):
            for _ in range(20):
                stacks = [rng.randint(1, 10000) for _ in range(size)]
                hero, villain = rng.sample(range(size), 2)
                pot = rng.randint(1, 10000)
                cost = rng.randint(1, stacks[hero])
                payouts = [100] + [0] * (size - 1)
                self.assertAlmostEqual(call_threshold(stacks, hero, villain, pot, cost, payouts), cost / (pot + cost), places=10)

    def test_allin_villain_and_previously_eliminated_seats(self):
        cases = [([100, 0, 150, 200], 0, 1, 180, 100, [50, 30, 20, 0]),
                 ([100, 0, 150, 0], 0, 2, 180, 100, [50, 30, 20, 0]),
                 ([100, 0], 0, 1, 400, 100, [100, 0])]
        for stacks, h, v, pot, cost, payouts in cases:
            self.assertAlmostEqual(call_threshold(stacks, h, v, pot, cost, payouts),
                                   brute_call_threshold(stacks, h, v, pot, cost, payouts), places=10)

    def test_icm_allin_risk_premium(self):
        stacks = [100, 100, 100, 100]
        threshold = call_threshold(stacks, 0, 1, 100, 100, [.5, .3, .2, 0])
        self.assertGreater(threshold, .5)
        self.assertLess(threshold, 1)

    def test_invalid_utility_inputs(self):
        for stacks, hero, payouts in (([1, 2], 0, [1]), ([1, 0], 0, [1, 0]),
                                     ([1, -1], 0, [1, 0]), ([1, 2], 2, [1, 0]),
                                     ([1, float('inf')], 0, [1, 0]),
                                     ([1, 2], 0, [0, 1]), ([1, 2], 0, [1, float('nan')]),
                                     ([1] * 9, 0, [1] * 9)):
            with self.assertRaises(ValueError):
                utility(stacks, hero, payouts)

    def test_invalid_call_resolution_inputs(self):
        for arguments in (([100, 100], 0, 0, 50, 20, [1, 0]),
                          ([100, 100], 0, 1, -1, 20, [1, 0]),
                          ([100, 100], 0, 1, 50, 101, [1, 0]),
                          ([100, -1], 0, 1, 50, 20, [1, 0]),
                          ([100, 100], 0, 1, 50, float('nan'), [1, 0])):
            with self.assertRaises(ValueError):
                call_threshold(*arguments)


if __name__ == "__main__":
    unittest.main()
