"""Independent five-card enumeration checks, not a copy of fast evaluation."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import random
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from equity import evaluate, _features, analyze, _MODEL, _FAST_PREDICT


def reference5(cards):
    rs = sorted((c // 4 + 2 for c in cards), reverse=True)
    groups = sorted(((n, r) for r, n in Counter(rs).items()), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    unique = set(rs)
    straight = max((r for r in range(14, 5-1, -1) if set(range(r-4,r+1)) <= unique), default=0)
    if {14,2,3,4,5} <= unique:
        straight = max(straight,5)
    if straight and flush:
        value = (8, straight)
    elif groups[0][0] == 4:
        value = (7, groups[0][1], groups[1][1])
    elif groups[0][0] == 3 and groups[1][0] == 2:
        value = (6, groups[0][1], groups[1][1])
    elif flush:
        value = (5,*rs)
    elif straight:
        value = (4,straight)
    elif groups[0][0] == 3:
        value = (3,groups[0][1], *sorted((r for n,r in groups[1:]),reverse=True))
    elif groups[0][0] == 2 and groups[1][0] == 2:
        value = (2,*sorted((groups[0][1],groups[1][1]),reverse=True),groups[2][1])
    elif groups[0][0] == 2:
        value = (1,groups[0][1],*sorted((r for n,r in groups[1:]),reverse=True))
    else:
        value = (0,*rs)
    return (value[0]<<20) + sum(r << (16-4*i) for i,r in enumerate(value[1:]))


class EvaluatorTests(unittest.TestCase):
    def test_all_categories_and_board_ties(self):
        hands = [[48,44,40,36,32], [48,49,50,51,44], [48,49,50,44,45],
                 [48,36,24,12,0], [48,0,5,10,15], [48,49,50,44,40],
                 [48,49,44,45,40], [48,49,44,40,36], [48,41,34,23,12]]
        for cards in hands:
            self.assertEqual(evaluate(cards), reference5(cards), cards)
        board = [48,44,40,36,32]
        self.assertEqual(evaluate([0,1]+board),evaluate([4,5]+board))

    def test_random_best_five_enumeration(self):
        rng = random.Random(1212001)
        for n, count in ((5,3000),(6,2000),(7,3000)):
            for _ in range(count):
                cards = rng.sample(range(52), n)
                self.assertEqual(evaluate(cards),max(map(reference5, combinations(cards,5))),cards)

    def test_model_finite_suit_invariant(self):
        rng = random.Random(1212002)
        for i in range(1000):
            cards=rng.sample(range(52),5+i%3)
            h,b=cards[:2],cards[2:]
            a=analyze(h,b)
            self.assertTrue(0<=a['equity_random']<=1)
            self.assertTrue(0<=a['equity_tight']<=1)
            perm=[2,0,3,1]
            remap=lambda cs:[4*(c//4)+perm[c%4] for c in cs]
            a2=analyze(remap(h),remap(b))
            self.assertAlmostEqual(a['equity_random'],a2['equity_random'],places=10)
            self.assertAlmostEqual(a['equity_tight'],a2['equity_tight'],places=10)
            self.assertEqual(_features(h,b),_features(remap(h),remap(b)))
            self.assertEqual(_features(h,b),_features(h[::-1],b))

    def test_locked_boards_and_draw_ownership(self):
        for board in ([48,44,40,36,32], [40,41,42,43,48], [48,49,50,51,44],
                      [48,45,42,39,32]):
            h=sorted(set(range(52))-set(board))[:2]
            self.assertEqual(analyze(h,board)['equity_random'],.5,board)
            self.assertTrue(analyze(h,board)['locked_split'])
        # Four clubs on the turn give no flush draw to two non-club cards.
        self.assertFalse(analyze([49,45],[0,12,24,36])['flush_draw'])
        self.assertTrue(analyze([48,45],[0,12,24])['flush_draw'])

    def test_exported_source_matches_tree_asset(self):
        if _MODEL is None or _FAST_PREDICT is None:
            self.skipTest('Optional trained assets absent')
        rng=random.Random(1212003)
        for i in range(200):
            cs=rng.sample(range(52),5+i%3);fs=_features(cs[:2],cs[2:]);ys=list(_MODEL['base'])
            for f,threshold,left,right,values in _MODEL['trees']:
                node=0
                while left[node]>=0:
                    node=left[node] if fs[f[node]]<=threshold[node] else right[node]
                ys=[a+b for a,b in zip(ys,values[node])]
            for a,b in zip(ys,_FAST_PREDICT(fs)):
                self.assertAlmostEqual(a,b,places=10)


if __name__ == '__main__': unittest.main()
