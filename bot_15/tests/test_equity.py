import itertools
import random
import unittest
from bot.evaluator import evaluate
from bot.equity import features, estimate, preflop_percentile, hand_class, forced_chop


def cards(s):
    ranks='23456789TJQKA';suits='cdhs'
    return [ranks.index(x[0])*4+suits.index(x[1]) for x in s.split()]


def reference5(cs):
    rs=sorted((c//4+2 for c in cs),reverse=True)
    groups=sorted(((rs.count(r),r) for r in set(rs)),reverse=True)
    flush=len({c%4 for c in cs})==1
    unique=sorted(set(rs));straight=0
    if len(unique)==5:
        if unique[-1]-unique[0]==4:straight=unique[-1]
        elif unique==[2,3,4,5,14]:straight=5
    if flush and straight:return (8,straight)
    if groups[0][0]==4:return (7,groups[0][1],groups[1][1])
    if [n for n,r in groups]==[3,2]:return (6,groups[0][1],groups[1][1])
    if flush:return (5,*rs)
    if straight:return (4,straight)
    if groups[0][0]==3:return (3,*[r for n,r in groups])
    if [n for n,r in groups[:2]]==[2,2]:return (2,*[r for n,r in groups])
    if groups[0][0]==2:return (1,*[r for n,r in groups])
    return (0,*rs)


def encode(rank):
    return rank[0]*759375+sum(r*15**(4-i) for i,r in enumerate(rank[1:]))


class EquityTests(unittest.TestCase):
    def test_named_hands(self):
        hands=['As Ks Qs Js Ts 2h 2d','Ah Ad Ac As Kh Kd Qs','Ah Ad Ac Kh Kd Ks 2c','Ah Jh 9h 5h 3h 2h Kd','Ah 2d 3c 4s 5h Kc Qd','Ah Ad Ac Kh Qd Js 2s','Ah Ad Kh Kd Qs Qh 2d','Ah Ad Kh Qd Js 9c 2s','Ah Kd Qs Jc 9h 5c 2s']
        ranks=[evaluate(cards(s)) for s in hands]
        self.assertEqual([v//759375 for v in ranks],list(range(8,-1,-1)))
        self.assertEqual(ranks,sorted(ranks,reverse=True))
    def test_random_against_independent_five_card_reference(self):
        rng=random.Random(89215)
        for n in (5,6,7):
            for _ in range(1200):
                cs=rng.sample(range(52),n)
                ref=max(reference5(x) for x in itertools.combinations(cs,5))
                self.assertEqual(evaluate(cs),encode(ref),cs)
    def test_board_plays_tie(self):
        b=cards('As Ks Qs Js Ts')
        self.assertEqual(evaluate(b+cards('2c 3d')),evaluate(b+cards('9s 9d')))
    def test_flush_uses_top_five(self):
        self.assertEqual(evaluate(cards('Ac Jc 9c 7c 5c 3c 2c')),evaluate(cards('Ac Jc 9c 7c 5c')))
    def test_features_suit_and_order_invariance(self):
        rng=random.Random(9815)
        for _ in range(300):
            cs=rng.sample(range(52),rng.choice((5,6,7)));h,b=cs[:2],cs[2:]
            f=features(h,b)
            self.assertEqual(f,features(h[::-1],b[::-1]))
            suits=rng.sample(range(4),4)
            m=lambda c:(c//4)*4+suits[c%4]
            self.assertEqual(f,features(list(map(m,h)),list(map(m,b))))
    def test_preflop(self):
        self.assertLess(preflop_percentile(cards('Ac Ad')),preflop_percentile(cards('7c 2d')))
        self.assertGreater(estimate(cards('Ac Ad'),[]),.83)
        self.assertLess(estimate(cards('7c 2d'),[]),.40)
        self.assertNotEqual(hand_class(cards('Ac Kc')),hand_class(cards('Ac Kd')))
    def test_estimate_bounds(self):
        rng=random.Random(85)
        for _ in range(50):
            cs=rng.sample(range(52),rng.choice((5,6,7)))
            for r in (.03,.2,.5,1):self.assertTrue(0<=estimate(cs[:2],cs[2:],r)<=1)
    def test_exact_outcomes(self):
        self.assertEqual(estimate(cards('As Ks'),cards('Qs Js Ts 2h 3c')) ,1)
        self.assertEqual(estimate(cards('2h 3c'),cards('As Ks Qs Js Ts')), .5)
        self.assertEqual(estimate(cards('2h 3c'),cards('As Kd Qh Js Tc')), .5)
        self.assertEqual(estimate(cards('2h 3c'),cards('7s 7h 7d 7c Ac')), .5)
        self.assertEqual(estimate(cards('2h 3c'),cards('As Ah Ad Ac Kc')), .5)
    def test_forced_chop(self):
        for b in ('As Ks Qs Js Ts','As Kd Qh Js Tc','As Ah Ad Ac Kc','7s 7h 7d 7c Ac'):
            self.assertTrue(forced_chop(cards(b)))
        for b in ('As Ks Qs Js 9s','As Ah Ad Ac Qc','7s 7h 7d 7c Kc','As Ks Qs Jh Td','As Ah Ad Kc Kd'):
            self.assertFalse(forced_chop(cards(b)))


if __name__=='__main__':unittest.main()
