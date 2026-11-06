import random
import unittest
from bot.evaluator import evaluate
from training.features_v3 import make_features, FEATURE_COUNT


class ThreatFeaturesTests(unittest.TestCase):
    def test_threat_upper_category_by_exhaustion(self):
        rng = random.Random(161116)
        for n in (3, 4, 5):
            for _ in range(12):
                cards = rng.sample(range(52), n + 2)
                h, b = cards[:2], cards[2:]
                f = make_features(h, b)
                self.assertEqual(len(f), FEATURE_COUNT)
                unseen = [c for c in range(52) if c not in cards]
                strongest = max(evaluate(b + [a, z]) for i, a in enumerate(unseen) for z in unseen[i + 1:])
                self.assertEqual(f[72], strongest >> 20)
                if strongest >> 20 == 8:
                    self.assertEqual(f[67], (strongest >> 16) & 15)
                elif strongest >> 20 == 7:
                    self.assertEqual(f[68], (strongest >> 16) & 15)
                elif strongest >> 20 == 6:
                    self.assertEqual(f[70], (strongest >> 16) & 15)
                    self.assertEqual(f[71], (strongest >> 12) & 15)
                elif strongest >> 20 == 4:
                    self.assertEqual(f[64], (strongest >> 16) & 15)

    def test_rare_fullhouse_upgrade_is_explicit(self):
        # Board 22233; private3 promotes the best full house to33322.
        f = make_features([20, 5], [2, 1, 0, 6, 4], 2, .25, 0)
        self.assertEqual(f[6], 6)
        self.assertEqual(f[12], 6)
        self.assertEqual(f[55], 1)  # Best hand improves the board.
        self.assertEqual(f[57], 1)  # Trip rank is one higher than board trip.


if __name__ == '__main__':
    unittest.main()
