import hashlib
import json
import random
import unittest
from pathlib import Path

from bot3._equity_fast import MODEL_SHA256, predict
from bot3.cards import features, hand_class
from bot3.equity import EquityModel


class ModelTests(unittest.TestCase):
    def test_generated_model_fingerprint_and_predictions(self):
        raw = Path("assets/equity_model.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), MODEL_SHA256)
        model = json.loads(raw)
        rng = random.Random(907)
        for _ in range(500):
            nb = rng.choice((3, 4, 5))
            cs = rng.sample(range(52), nb + 2)
            x = features(cs[:2], cs[2:])
            expected = model["base"][:]
            for t in model["trees"]:
                n = t[0]
                while n[0] >= 0:
                    n = t[n[2] if x[n[0]] <= n[1] else n[3]]
                for i in range(6):
                    expected[i] += n[4 + i]
            for a, b in zip(predict(x), expected):
                self.assertAlmostEqual(a, b, places=12)

    def test_preflop_sanity_and_pushfold(self):
        m = EquityModel()
        aa, junk = hand_class(48, 49), hand_class(0, 21)
        self.assertLess(m.percentile[aa], .01)
        self.assertGreater(m.preflop((48, 49)), .84)
        self.assertLess(m.preflop((0, 21)), .36)
        for d in m.pushfold["depths"]:
            self.assertGreater(m.heads_up((48, 49), d, "push"), .99)
            self.assertGreater(m.heads_up((48, 49), d, "call"), .99)
        self.assertLess(max(m.pushfold["exploitability_bb_per_hand"]), .001)
        self.assertLess(m.heads_up((0, 21), 12, "call"), .01)
