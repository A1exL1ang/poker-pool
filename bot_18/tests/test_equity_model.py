import unittest
import random
import json
from pathlib import Path
from equity.model import estimate, describe, forced_tie, estimate_premium, MODEL_AVAILABLE, PREFLOP_AVAILABLE, PREMIUM_AVAILABLE
from tests.test_equity_evaluator import cards


class ModelTests(unittest.TestCase):
    def test_forced_ties_and_royal_nuts(self):
        for n in range(1,8):
            self.assertEqual(estimate(cards("2c 3d"),cards("As Ks Qs Js Ts"),.12,n),1/(n+1))
            self.assertEqual(estimate(cards("2c 3d"),cards("Ac Ad Ah As Ks"),.12,n),1/(n+1))
            self.assertEqual(estimate(cards("2c 3d"),cards("7c 7d 7h 7s As"),.12,n),1/(n+1))
            self.assertEqual(estimate(cards("2c 3d"),cards("Ac Kd Qh Js Tc"),.12,n),1/(n+1))
            self.assertEqual(estimate(cards("As Ks"),cards("Qs Js Ts 2h 3d"),.12,n),1)
        self.assertTrue(forced_tie(cards("2c 3d"),cards("Ac Kd Qh Js Tc")))
        self.assertFalse(forced_tie(cards("2c 3d"),cards("Ac Kc Qc Js Td")))
        self.assertFalse(forced_tie(cards("2c 3d"),cards("Ac Ad Ah As Qs")))

    def test_feature_symmetry(self):
        self.assertEqual(describe(cards("Ac Kd"),cards("7c 4d 2s")),
                         describe(cards("Kd Ac"),cards("2s 7c 4d")))

    @unittest.skipUnless(PREFLOP_AVAILABLE,"trained preflop asset not present")
    def test_preflop_sanity(self):
        aa=estimate(cards("Ac Ad"),[],1,1)
        self.assertGreater(aa,.83); self.assertLess(aa,.87)
        for text in ("Ac Ad","Ac Kd","7c 2d","2c 2d"):
            values=[estimate(cards(text),[],1,n) for n in range(1,8)]
            for a,b in zip(values,values[1:]):self.assertGreater(a,b)

    @unittest.skipUnless(PREMIUM_AVAILABLE,"trained premium asset not present")
    def test_premium_sanity(self):
        self.assertGreater(estimate_premium(cards("Ac Ad"),True),.74)
        self.assertLess(estimate_premium(cards("Kc Kd"),True),.26)
        self.assertLess(estimate_premium(cards("Jc Jd"),False),.40)
        self.assertGreater(estimate_premium(cards("Ac Ad"),False),.78)

    @unittest.skipUnless(MODEL_AVAILABLE,"trained equity asset not present")
    def test_bounded_and_multiway_monotonic(self):
        for text in ("Ac Kd 7c 4d 2s", "Ac Ad 7c 4d 2s", "8c 9c 6c 7c 2s", "Ac 2d 3c 4h 5s"):
            hand=cards(text)
            values=[estimate(hand[:2],hand[2:],.25,n) for n in range(1,8)]
            for a in values:self.assertTrue(0<=a<=1)
            for a,b in zip(values,values[1:]):self.assertGreaterEqual(a+1e-12,b)

    @unittest.skipUnless(MODEL_AVAILABLE,"trained equity asset not present")
    def test_untrained_counts_are_not_aliased_to_fewer_opponents(self):
        hole,board=cards('Ac Ad'),cards('7c 4d 2s')
        values={n:estimate(hole,board,1,n) for n in (3,4,5,6,7)}
        self.assertGreater(values[3],values[4])
        self.assertGreater(values[4],values[5])
        self.assertGreater(values[5],values[6])
        self.assertGreater(values[6],values[7])

    @unittest.skipUnless(MODEL_AVAILABLE,"trained equity asset not present")
    def test_compiled_matches_interpreter(self):
        from equity.model import _COMPILED
        if _COMPILED is None:self.skipTest("compiled asset not present")
        model=json.loads((Path(__file__).resolve().parents[1]/"equity"/"model.json").read_text())
        from training.fit import predict_export
        rng=random.Random(182440)
        for i in range(500):
            hand=rng.sample(range(52),5+i%3)
            row=(*describe(hand[:2],hand[2:]),(.06,.12,.25,.45,1)[i%5],i%7+1)
            self.assertAlmostEqual(max(0,min(1,_COMPILED(row))),predict_export(model,row),places=12)


if __name__=="__main__":unittest.main()
