import random
import unittest
from pokerbot.native import LIB
from pokerbot.evaluator import evaluate_bits,packed
from pokerbot.equity import EquityEstimator
from pokerbot.equity import COMBOS
from pokerbot.icm import hero_icm,hero_icm_python


@unittest.skipUnless(LIB is not None,"optional native helper not built for this platform")
class NativeTests(unittest.TestCase):
    def test_native_evaluation(self):
        rng=random.Random(14517)
        for i in range(30000):
            hand=rng.sample(range(52),5+i%3)
            bits=packed(hand)
            self.assertEqual(evaluate_bits(bits),LIB.bot_evaluate(bits))

    def test_sample_for_sample_equivalence(self):
        rng=random.Random(2345)
        for i in range(500):
            n=1+i%7
            street=(3,4,5)[i%3]
            hand=rng.sample(range(52),street+2)
            fractions=[rng.uniform(.05,1) for _ in range(n)]
            a,b=EquityEstimator(i+1),EquityEstimator(i+1)
            e=a.estimate(hand[:2],hand[2:],fractions,.7,trials=32)
            f=b.estimate_python(hand[:2],hand[2:],fractions,.7,trials=32)
            self.assertAlmostEqual(e[0],f[0],places=12)
            self.assertAlmostEqual(e[1],f[1],places=12)
            self.assertEqual(a.seed,b.seed)

    def test_icm_native_matches_python(self):
        rng=random.Random(749)
        for n in range(2,9):
            for _ in range(50):
                stacks=[rng.randrange(1,2000) for i in range(n)]
                payouts=sorted([rng.randrange(0,100) for i in range(n)],reverse=True)
                for h in range(n):
                    self.assertAlmostEqual(hero_icm(stacks,h,payouts),hero_icm_python(stacks,h,payouts),places=10)

    def test_weighted_layers_match_python(self):
        rng=random.Random(514917)
        for i in range(150):
            n=2+i%6
            street=(0,3,4,5)[i%4]
            cs=rng.sample(range(52),street+2)
            fractions=[.2+.06*j for j in range(n)]
            layers=[((1<<n)-1,.3),((1<<(n-1))-1,.7)]
            a,b=EquityEstimator(i+90),EquityEstimator(i+90)
            ea=a.estimate(cs[:2],cs[2:],fractions,0,trials=32,layers=layers)
            eb=b.estimate_python(cs[:2],cs[2:],fractions,0,trials=32,layers=layers)
            self.assertAlmostEqual(ea[0],eb[0],places=12)
            self.assertEqual(a.seed,b.seed)

    def test_exact_river_uniform_range(self):
        rng=random.Random(91217)
        for i in range(60):
            cs=rng.sample(range(52),7)
            hole,board=cs[:2],cs[2:]
            h,b=packed(hole),packed(board)
            score=evaluate_bits(h|b)
            fraction=(.1,.3,.6,1.)[i%4]
            limit=max(66,int(fraction*1326))
            wins=count=0.
            for _,_,vb in COMBOS[:limit]:
                if vb&(h|b):
                    continue
                vs=evaluate_bits(vb|b)
                wins+=1 if score>vs else .5 if score==vs else 0
                count+=1
            eq,error=EquityEstimator().estimate(hole,board,[fraction],0)
            self.assertAlmostEqual(eq,wins/count,places=12)
            self.assertEqual(error,0.)
