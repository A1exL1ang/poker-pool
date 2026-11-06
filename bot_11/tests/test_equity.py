"""Real-asset conformance, independent of fitting libraries."""
import random
import unittest
from pokerbot.equity import EquityModel, hand_key, load_preflop
from pokerbot.features import features


class EquityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = EquityModel()

    def test_compact_fast_path_matches_reference_traversal(self):
        rng = random.Random(11777)
        for i in range(150):
            cards = rng.sample(range(52), 2+(3,4,5)[i%3])
            x = features(cards[:2], cards[2:])
            sums = [0]*10
            for fs, ts, links, values in self.model.trees:
                n = 0
                while fs[n] >= 0:
                    n = n+1 if x[fs[n]] <= ts[n] else links[n]
                base = (-links[n]-1)*10
                for j in range(10):
                    sums[j] += values[base+j]
            expected = [s/(255*len(self.model.trees)) for s in sums]
            actual = self.model.predict(x)
            for a,b in zip(actual,expected):
                self.assertAlmostEqual(a,b,places=14)
                self.assertTrue(0 <= a <= 1)

    def test_preflop_asset_completeness_and_equity_sanity(self):
        rows = load_preflop()
        self.assertEqual(len(rows),169)
        self.assertTrue(.84 < rows['AA']['equity'][0] < .87)
        self.assertTrue(.80 < rows['KK']['equity'][0] < .84)
        self.assertTrue(.30 < rows['72o']['equity'][0] < .40)
        self.assertLess(rows['AA']['percentile'], rows['KK']['percentile'])
        self.assertEqual(hand_key([48,49]),'AA')
        self.assertEqual(hand_key([48,44]),'AKs')
        self.assertEqual(hand_key([48,45]),'AKo')
        for r in rows.values():
            self.assertTrue(0 <= r['percentile'] <= 1)
            self.assertEqual(len(r['equity']),7)
            self.assertTrue(all(0 <= v <= 1 for v in r['equity']))
            self.assertLessEqual(r['uniform3'],r['uniform2'])
            self.assertLessEqual(r['uniform2'],r['equity'][0])


if __name__ == '__main__':
    unittest.main()
