"""Rule-critical evaluator verification independent of model training labels."""
import random
import unittest

from equity.evaluator import evaluate, evaluate_reference, decode
from equity.features import FEATURE_COUNT, features
from equity.preflop import class_index, class_hole, preflop_percentile


def cards(text):
    ranks="23456789TJQKA";suits="cdhs"
    return [4*ranks.index(t[0])+suits.index(t[1]) for t in text.split()]


class EvaluatorTests(unittest.TestCase):
    def test_categories_and_ties(self):
        cases = [("Ac Kd Jh 9s 7c",0), ("Ac Ad Kh Qs 9c",1),
                 ("Ac Ad Kh Ks 9c",2),("Ac Ad Ah Ks 9c",3),
                 ("Ac 2d 3h 4s 5c",4),("Ac Jc 9c 7c 2c",5),
                 ("Ac Ad Ah Ks Kc",6),("Ac Ad Ah As Kc",7),
                 ("Ac Kc Qc Jc Tc",8)]
        values=[]
        for text,expected in cases:
            value=evaluate(cards(text));self.assertEqual(value>>20,expected)
            self.assertEqual(value,evaluate_reference(cards(text)));values.append(value)
        self.assertEqual(values,sorted(values))
        self.assertEqual(evaluate(cards("Ac Kc Qc Jc Tc 2d 3s")),evaluate(cards("Ac Kc Qc Jc Tc 9d 8s")))

    def test_difficult_seven_card_cases(self):
        for text in ("Ac Ad Ah Kc Kd Kh 2c", "Ac Ad Kc Kd Qc Qd 2c",
                     "Ac 2c 3c 4c 5c 6c 7c", "Ac Ad Ah As Kc Kd Kh",
                     "2c 3c 4c 5c 7c 6d Ac", "2c 3d 4h 5s Ac Kc Qc"):
            self.assertEqual(evaluate(cards(text)),evaluate_reference(cards(text)))

    def test_random_independent_reference(self):
        rng=random.Random(180930)
        for n in (5,6,7):
            for _ in range(2000):
                hand=rng.sample(range(52),n)
                self.assertEqual(evaluate(hand),evaluate_reference(hand),hand)

    def test_suit_and_order_invariance(self):
        rng=random.Random(180931)
        for _ in range(1000):
            hand=rng.sample(range(52),7); suits=rng.sample(range(4),4)
            transformed=[4*(c//4)+suits[c%4] for c in hand]
            rng.shuffle(transformed)
            self.assertEqual(evaluate(hand),evaluate(transformed))
            a=features(hand[:2],hand[2:]);b=features(transformed[:2],transformed[2:])
            self.assertEqual(len(a),FEATURE_COUNT)
            # Preserve hole/board boundaries for feature invariance.
            h=[4*(c//4)+suits[c%4] for c in hand[:2]]
            board=[4*(c//4)+suits[c%4] for c in hand[2:]]
            self.assertEqual(a,features(h,board))
            self.assertEqual(a,features(list(reversed(h)),list(reversed(board))))

    def test_preflop_classes(self):
        counts=[0]*169
        for a in range(52):
            for b in range(a+1,52):
                counts[class_index((a,b))]+=1
        for i in range(169):
            a,b=divmod(i,13)
            self.assertEqual(counts[i],6 if a==b else 4 if a>b else 12)
            self.assertEqual(class_index(class_hole(i)),i)
        self.assertLess(preflop_percentile(cards("Ac Ad")),preflop_percentile(cards("7c 2d")))


if __name__=="__main__":
    unittest.main()
