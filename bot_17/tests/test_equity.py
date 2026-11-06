import ctypes
from itertools import combinations
from math import comb
import random
import unittest

import equity

def reference_five(cards):
    """Independent simple five-card tuple evaluator used only in tests."""
    values = sorted((c//4+2 for c in cards), reverse=True)
    groups = sorted(((values.count(v),v) for v in set(values)), reverse=True)
    flush = len({c%4 for c in cards}) == 1
    unique = sorted(set(values))
    straight = 5 if unique == [2,3,4,5,14] else unique[-1] if len(unique)==5 and unique[-1]-unique[0]==4 else 0
    if flush and straight:
        result = (8,straight)
    elif groups[0][0]==4:
        result = (7,groups[0][1],groups[1][1])
    elif [g[0] for g in groups]==[3,2]:
        result = (6,groups[0][1],groups[1][1])
    elif flush:
        result = (5,*values)
    elif straight:
        result = (4,straight)
    elif groups[0][0]==3:
        result = (3,groups[0][1],*sorted((g[1] for g in groups[1:]),reverse=True))
    elif [g[0] for g in groups][:2]==[2,2]:
        result = (2,*sorted((groups[0][1],groups[1][1]),reverse=True),groups[2][1])
    elif groups[0][0]==2:
        result = (1,groups[0][1],*sorted((g[1] for g in groups[1:]),reverse=True))
    else:
        result = (0,*values)
    return (result[0]<<20) + sum(v<<(16-4*i) for i,v in enumerate(result[1:]))

def cards(text):
    ranks='23456789TJQKA'; suits='cdhs'
    return [ranks.index(c[0])*4+suits.index(c[1]) for c in text.split()]

class EquityTests(unittest.TestCase):
    def test_categories_and_wheel(self):
        cases = [
            ('Ac Kd 9s 7h 3c',0), ('Ac Ad 9s 7h 3c',1),
            ('Ac Ad 9s 9h 3c',2), ('Ac Ad As 7h 3c',3),
            ('Ac 2d 3s 4h 5c',4), ('Ac Jc 9c 7c 3c',5),
            ('Ac Ad As 7h 7c',6), ('Ac Ad As Ah 3c',7),
            ('Ac Kc Qc Jc Tc',8),
        ]
        for text,cat in cases:
            c=cards(text)
            self.assertEqual(equity.category(c),cat)
            self.assertEqual(equity.evaluate(c),reference_five(c))
        self.assertGreater(equity.evaluate(cards('2c 3d 4s 5h 6c')),equity.evaluate(cards('Ac 2d 3s 4h 5c')))

    def test_independent_differential(self):
        rng=random.Random(178271)
        for n in (5,6,7):
            for _ in range(1800):
                c=rng.sample(range(52),n)
                expected=max(reference_five(hand) for hand in combinations(c,5))
                self.assertEqual(equity.evaluate_python(c),expected,c)
                self.assertEqual(equity.evaluate(c),expected,c)

    def test_awkward_groups_and_board_ties(self):
        for text in ('Ac Ad As Kc Kd Ks 2c','Ac Ad Kc Kd Qc Qd 2c','Ac Ad As Ah Kc Kd Qc',
                     'Ac 2c 3c 4c 5c 6c 7c','Ac Kd Qc Js Tc 2c 3d'):
            c=cards(text)
            self.assertEqual(equity.evaluate(c),max(reference_five(hand) for hand in combinations(c,5)))
        board=cards('Ac Kc Qc Jc Tc')
        for opponents in (1,3,7):
            self.assertAlmostEqual(equity.estimate(cards('2d 3d'),board,opponents),1/(opponents+1),places=12)
        board=cards('Ac Ad Ah As Kc')
        for opponents in (1,3,7):
            self.assertAlmostEqual(equity.estimate(cards('2d 3d'),board,opponents),1/(opponents+1),places=12)
            self.assertAlmostEqual(equity.model_estimate(cards('2d 3d'),board,opponents,.8,.9),1/(opponents+1),places=12)
        board=cards('Ac Kd Qh Js Tc')
        for opponents in (1,3,7):
            self.assertAlmostEqual(equity.estimate(cards('2d 3d'),board,opponents),1/(opponents+1),places=12)
            self.assertAlmostEqual(equity.model_estimate(cards('2d 3d'),board,opponents,.8,.9),1/(opponents+1),places=12)

    def test_sidepot_weighted_ties(self):
        h=cards('2d 3d');b=cards('Ac Kc Qc Jc Tc')
        expected=(30/3+3980/2)/4010
        self.assertAlmostEqual(equity.estimate(h,b,2,pots=[(30,3),(3980,2)]),expected,places=12)
        self.assertEqual(equity.estimate(h,b,2,pots=[(30,0)]),1.)
        self.assertEqual(equity.estimate(h,b,2,pots=[]),0.)
        self.assertAlmostEqual(equity.estimate(h,b,1,pots=[(20,0),(80,1)]),.6,places=12)

    def test_sidepot_full_mask_and_uncontested(self):
        h=cards('Ac Jd');b=cards('9h 5c 3s')
        ranges=[(.3,.5),(.9,.7)]
        base=equity.estimate(h,b,ranges=ranges,seed=7189)
        self.assertEqual(equity.estimate(h,b,ranges=ranges,seed=7189,pots=[(100,3)]),base)
        self.assertAlmostEqual(equity.estimate(h,b,ranges=ranges,seed=7189,pots=[(30,0),(70,3)]),.3+.7*base,places=12)
        with self.assertRaises(ValueError):equity.estimate(h,b,2,pots=[(10,4)])
        with self.assertRaises(ValueError):equity.estimate(h,b,2,pots=[(-1,3)])

    @unittest.skipUnless(equity._NATIVE is not None,'optional native accelerator absent')
    def test_all_five_card_category_counts(self):
        output=(ctypes.c_uint64*9)()
        equity._NATIVE.poker_five_counts.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
        equity._NATIVE.poker_five_counts(output)
        expected=[(comb(13,5)-10)*(4**5-4),13*comb(4,2)*comb(12,3)*4**3,
            comb(13,2)*comb(4,2)**2*11*4,13*4*comb(12,2)*4**2,10*(4**5-4),
            4*(comb(13,5)-10),13*4*12*6,13*12*4,40]
        self.assertEqual(list(output),expected)
        self.assertEqual(sum(output),comb(52,5))

    def test_nuts_and_determinism(self):
        h=cards('Ac Kc'); b=cards('Qc Jc Tc')
        self.assertEqual(equity.estimate(h,b,7,seed=991),equity.estimate(h,b,7,seed=991))
        self.assertGreater(equity.estimate(h,b,7,seed=991),.99)

    @unittest.skipUnless(equity._NATIVE is not None,'optional native accelerator absent')
    def test_river_vs_exact_enumeration(self):
        h=cards('Ac Jd'); b=cards('As 9h 5c 3s 2d')
        deck=[c for c in range(52) if c not in h+b]
        rank=equity.evaluate(h+b); exact=0
        for opp in combinations(deck,2):
            other=equity.evaluate(list(opp)+b)
            exact += 1 if rank>other else .5 if rank==other else 0
        exact /= comb(len(deck),2)
        sampled=equity.native_equity(h,b,samples=40000,seed=77771)
        self.assertLess(abs(exact-sampled),.014)
        if hasattr(equity._NATIVE,'poker_river_equity'):
            self.assertAlmostEqual(equity.estimate(h,b),exact,places=12)
            self.assertEqual(equity.estimate(h,b,seed=1),equity.estimate(h,b,seed=991))

    def test_input_validation(self):
        with self.assertRaises(ValueError):equity.evaluate([0,0,1,2,3])
        with self.assertRaises(ValueError):equity.evaluate([0,1,2,3,52])
        with self.assertRaises(ValueError):equity.estimate([0,1],[2,3,1])

    def test_range_and_class(self):
        self.assertEqual(equity.hand_class(cards('Ac Ad')),'AA')
        self.assertEqual(equity.hand_class(cards('Ac Kc')),'AKs')
        self.assertGreater(equity.percentile(cards('Ac Ad')),equity.percentile(cards('2c 7d')))
        for opponents in range(1,8):
            for strength in (0.,.5,1.):
                self.assertTrue(0 < equity.preflop(cards('Ac Ad'),opponents,strength)<1)

    @unittest.skipUnless(equity._MODEL is not None and equity._COMPILED_PREDICTOR is not None,'optional compiled model absent')
    def test_compiled_model_matches_json(self):
        rng=random.Random(170817)
        for _ in range(100):
            c=rng.sample(range(52),7);nb=rng.choice((3,4,5));n=rng.randint(1,7)
            f=equity.model_features(c[:2],c[2:2+nb],n,rng.random(),rng.random())
            expected=equity._MODEL['base']
            for tree in equity._MODEL['trees']:
                index=0
                while tree[index][0]>=0:
                    node=tree[index];index=node[2] if f[node[0]]<node[1] else node[3]
                expected+=equity._MODEL['rate']*tree[index][4]
            self.assertAlmostEqual(equity._COMPILED_PREDICTOR(f),expected,places=12)

if __name__=='__main__':unittest.main()
