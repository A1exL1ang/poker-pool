import unittest
import random

import equity_model
from poker_eval import evaluate, features
from equity_features import DERIVED_NAMES, board_plays, derived_features, extend_features


def cards(text):
    return ["23456789TJQKA".index(c[0])*4+"cdhs".index(c[1]) for c in text.split()]


class EquityTests(unittest.TestCase):
    def test_canonical_class_mapping(self):
        self.assertEqual(equity_model.preflop_class(cards("Ac Ad")), 168)
        self.assertEqual(equity_model.preflop_class(cards("Ac Kc")), 167)
        self.assertEqual(equity_model.preflop_class(cards("Ac Kd")), 155)
        self.assertEqual(equity_model.preflop_class(cards("2h 2s")), 0)
        for text in ("Ac Ad", "Ac Kc", "Ac Kd", "2h 2s", "7d 6d"):
            h = cards(text)
            self.assertEqual(equity_model.preflop_class(h), equity_model.preflop_class(h[::-1]))
            rotated = [4*(c//4)+(c%4+1)%4 for c in h]
            self.assertEqual(equity_model.preflop_class(h), equity_model.preflop_class(rotated))

    def test_bounded_equity_and_single_feature_pass(self):
        for h, b in (("Ac Ad", "2c 7d Th"), ("Ac Kc", "Qc Jc Tc 4d 5h"),
                     ("2c 3d", "Ah Kh Qh Jh Th"), ("9d 8c", "9c 6h 2d Ts")):
            hole, board = cards(h), cards(b)
            for w, a in ((.05, 1), (.35, .5), (1, 0)):
                f = features(hole, board, w, a)
                p = equity_model.predict_from_features(f)
                self.assertGreaterEqual(p, 0)
                self.assertLessEqual(p, 1)
                self.assertEqual(p, equity_model.predict_equity(hole, board, w, a))

    def test_preflop_bounds_and_loaded_strength(self):
        for hole in (cards("Ac Ad"), cards("7c 2d"), cards("Ac Kc")):
            self.assertGreaterEqual(equity_model.preflop_percentile(hole), 0)
            self.assertLessEqual(equity_model.preflop_percentile(hole), 1)
            for w in (-1, .05, .1, .35, 1, 2):
                for n in range(1, 8):
                    p = equity_model.preflop_equity(hole, w, n)
                    self.assertGreater(p, 0)
                    self.assertLess(p, 1)
        if equity_model.asset_status()["preflop_table"]:
            self.assertLess(equity_model.preflop_percentile(cards("Ac Ad")), .01)
            self.assertGreater(equity_model.preflop_equity(cards("Ac Ad")), .83)
            self.assertLess(equity_model.preflop_equity(cards("7c 2d")), .37)

    def test_multiway_probability_and_monotonicity(self):
        for h,b in (("Ah Kh","Qh Jh 2c"),("Ac Ad","2c 7d Th 5h"),("5c 4d","9c 8d 7h 2s Ac")):
            f=features(cards(h),cards(b))
            for q in (0,.05,.2,.5,.8,1):
                previous=q
                for n in (1,1.5,2,3,4.5,5,6,7):
                    value=equity_model.multiway_equity(q,f,n)
                    self.assertGreaterEqual(value,0)
                    self.assertLessEqual(value,previous+1e-12)
                    previous=value

    def test_derived_board_rank_relationships(self):
        f=features(cards("8c Ad"),cards("7d 6h 5s 4c 3d"))
        d=dict(zip(DERIVED_NAMES,derived_features(f)))
        self.assertEqual(d["straight_board_gain"],1)
        self.assertEqual(d["straight_nut_gap"],-1)
        self.assertEqual(len(extend_features(f)),55)
        f=features(cards("Qd 7c"),cards("9h 9c 7d 5s 5c"))
        d=dict(zip(DERIVED_NAMES,derived_features(f)))
        self.assertEqual(d["highest_board_pair"],9)
        self.assertEqual(d["lowest_board_pair"],5)
        self.assertEqual(d["secondary_rank_gain"],2)

    def test_board_plays_full_kickers(self):
        cases = [
            ("Ah Kc Qd 8s 6h", "2c 3d", "7c 2d"),
            ("Ah Ac Kd Qs 7h", "2c 3d", "Jc 2d"),
            ("Ah Ac Kd Ks 7h", "2c 3d", "Qc 2d"),
            ("Ah Ac Ad Ks 7h", "2c 3d", "Qc 2d"),
            ("9h 8c 7d 6s 5h", "2c 3d", "Tc 2d"),
            ("Ah Jh 8h 5h 2h", "3c 4d", "3h 4c"),
            ("Ah Ac Ad Ks Kh", "2c 3d", "As 2d"),
            ("9h 9c 9d 9s Kh", "2c 3d", "Ac 2d"),
            ("9h 8h 7h 6h 5h", "2c 3d", "Th 2d"),
        ]
        for b,playing,improving in cases:
            self.assertTrue(board_plays(features(cards(playing),cards(b))), (b,playing))
            self.assertFalse(board_plays(features(cards(improving),cards(b))), (b,improving))

    def test_board_plays_multiway_ties(self):
        f=features(cards("2c 3d"),cards("Ah Ac Ad Ks Kh"))
        for n in range(1,8):
            self.assertAlmostEqual(equity_model.multiway_equity(.5,f,n),1/(n+1))
            self.assertAlmostEqual(equity_model.multiway_equity(.8,f,n),1/(n+1))
        self.assertAlmostEqual(equity_model.multiway_equity(.4,f,1),.4)
        self.assertGreater(equity_model.multiway_equity(.48,f,3),.2)

    def test_board_plays_matches_full_evaluator(self):
        rng=random.Random(190555)
        for _ in range(5000):
            h=rng.sample(range(52),7)
            f=features(h[:2],h[2:])
            self.assertEqual(board_plays(f),evaluate(h)==evaluate(h[2:]))


if __name__ == "__main__":
    unittest.main()
