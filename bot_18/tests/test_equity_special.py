import itertools
import unittest
from equity.model import estimate
from equity.special import _uniform_kicker_share
from training.stress import make_ranges, exact, parse


class ExactRareBoardTests(unittest.TestCase):
    def test_combinatorial_ties_against_toy_deck_enumeration(self):
        # Distinct physical cards carrying only rank relation to hero kicker.
        for lower,equal,higher in ((4,2,2),(5,3,0),(2,2,4),(6,1,1)):
            ranks=[-1]*lower+[0]*equal+[1]*higher
            for opponents in (1,2,3):
                for playing in (False,True):
                    outcomes=[]
                    for cards in itertools.permutations(range(len(ranks)),2*opponents):
                        if any(ranks[c]>0 for c in cards):outcomes.append(0);continue
                        ties=opponents if playing else sum(max(ranks[cards[2*i]],ranks[cards[2*i+1]])==0 for i in range(opponents))
                        outcomes.append(1/(ties+1))
                    target=sum(outcomes)/len(outcomes)
                    self.assertAlmostEqual(_uniform_kicker_share(lower,equal,higher,opponents,playing),target,places=11)

    def test_board_quads_exact_heads_up_ranges(self):
        ranges=make_ranges()
        for hole,board in (("Qd Kc","Ac Ad Ah As 2c"),("5h 3c","Ac Ad Ah As 2c"),
                           ("As 5s","7c 7d 7h 7s 9c"),("2s Ts","7c 7d 7h 7s 9c")):
            h,b=parse(hole),parse(board)
            for fraction in (.06,.12,.25,.45,1.0):
                self.assertAlmostEqual(estimate(h,b,fraction,1),exact(h,b,ranges[fraction]),places=12)

    def test_board_straights_exact_heads_up_ranges(self):
        ranges=make_ranges()
        for hole,board in (("Qd 9c","4c 5d 6h 7s 8c"),("Kd 2c","4c 5d 6h 7s 8c"),
                           ("7d 6s","Ac 2d 3h 4s 5c"),("2s 3s","Ac Kd Qh Js Tc")):
            h,b=parse(hole),parse(board)
            for fraction in (.06,.12,.25,.45,1.0):
                self.assertAlmostEqual(estimate(h,b,fraction,1),exact(h,b,ranges[fraction]),places=12)


if __name__=="__main__":unittest.main()
