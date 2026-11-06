"""Physical-card checks independent of the compact range implementation."""
import itertools
import unittest
from pokerbot.preflop import TABLES, class_id, legal_counts, representative


def physical_class(a, b):
    ranks = sorted((a // 4, b // 4), reverse=True)
    high, low = 12 - ranks[0], 12 - ranks[1]
    if high == low:
        return 13 * high + low
    return 13 * high + low if a % 4 == b % 4 else 13 * low + high


class PreflopTests(unittest.TestCase):
    def test_all_physical_classes(self):
        counts = [0] * 169
        for a, b in itertools.combinations(range(52), 2):
            expected = physical_class(a, b)
            self.assertEqual(class_id(a, b), expected)
            self.assertEqual(class_id(b, a), expected)
            counts[expected] += 1
        for i, count in enumerate(counts):
            row, col = divmod(i, 13)
            self.assertEqual(count, 6 if row == col else 4 if row < col else 12)
            self.assertEqual(class_id(*representative(i)), i)

    def test_legal_counts_against_physical_enumeration(self):
        for i in range(169):
            blocked = set(representative(i))
            expected = [0] * 169
            for a, b in itertools.combinations((c for c in range(52) if c not in blocked), 2):
                expected[physical_class(a, b)] += 1
            self.assertEqual(legal_counts(i), expected)
            self.assertEqual(sum(expected), 1225)

    @unittest.skipUnless(TABLES.loaded, "measured preflop asset not present")
    def test_independent_asset_sanity(self):
        for i in range(169):
            self.assertAlmostEqual(TABLES.matrix[i][i], .5)
            for j in range(169):
                self.assertTrue(0 <= TABLES.matrix[i][j] <= 1)
                self.assertAlmostEqual(TABLES.matrix[i][j] + TABLES.matrix[j][i], 1, places=6)
        aa = class_id(48, 49)
        self.assertTrue(.83 < TABLES.uniform[aa][0] < .87)
        self.assertTrue(all(a > b for a, b in zip(TABLES.uniform[aa], TABLES.uniform[aa][1:])))
        self.assertAlmostEqual(TABLES.mass(aa, 1.), 1.)


if __name__ == "__main__":
    unittest.main()
