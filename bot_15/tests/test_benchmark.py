import unittest
from collections import Counter
from benchmarks.engine import play_table
from benchmarks.opponents import BaselineAgent
from benchmarks.tournament import candidate_seat


class BenchmarkTests(unittest.TestCase):
    def test_crossed_seating_balances_relative_position_and_absolute_pairs(self):
        pairs = Counter()
        positions = Counter()
        for k in range(64):
            observations = []
            play_table([BaselineAgent('fold', i) for i in range(8)], 8100001 + k,
                       max_hands=1, observer=lambda o, a, t: observations.append(o))
            seat, button = candidate_seat(k), observations[0]['button']
            pairs[seat, button] += 1
            positions[(seat - button) % 8] += 1
        self.assertEqual(len(pairs), 64)
        self.assertEqual(set(pairs.values()), {1})
        self.assertEqual(list(positions.values()), [8] * 8)


if __name__ == '__main__':
    unittest.main()
