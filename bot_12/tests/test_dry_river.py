from itertools import combinations
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from equity import evaluate,exact_dry_river,exact_river,hand_key,_PREFLOP
import equity

class DryRiverTests(unittest.TestCase):
    def test_rank_collapse_against_all_990_card_combinations(self):
        rng=random.Random(1212077);checked=0
        for _ in range(600):
            cards=rng.sample(range(52),7);hole,board=cards[:2],cards[2:]
            exact=exact_dry_river(hole,board)
            if exact is None:continue
            remain=sorted(set(range(52))-set(cards));score=evaluate(cards)
            wins=twins=nt=0
            for opp in combinations(remain,2):
                os=evaluate(opp+tuple(board));w=2 if score>os else 1 if score==os else 0
                wins+=w
                if _PREFLOP['order'][hand_key(opp)]<266:nt+=1;twins+=w
            self.assertAlmostEqual(exact[0],wins/1980,places=12,msg=cards)
            self.assertAlmostEqual(exact[1],twins/(2*nt),places=12,msg=cards)
            checked+=1
        self.assertGreater(checked,200)

    def test_rejects_flush_possible_boards(self):
        self.assertIsNone(exact_dry_river([48,49],[0,4,8,13,18]))

    def test_all_suit_textures_against_990_combinations(self):
        rng=random.Random(1212078)
        for maxsuit in (2,3,4,5):
            checked=0
            while checked<100:
                cards=rng.sample(range(52),7);hole,board=cards[:2],cards[2:]
                if max(sum((c&3)==s for c in board) for s in range(4))!=maxsuit:continue
                checked+=1;exact=exact_river(hole,board,with_ties=True)
                remain=sorted(set(range(52))-set(cards));score=evaluate(cards);wins=twins=nt=ties=tties=0
                for opp in combinations(remain,2):
                    os=evaluate(opp+tuple(board));w=2 if score>os else 1 if score==os else 0
                    wins+=w
                    ties+=int(w==1)
                    if _PREFLOP['order'][hand_key(opp)]<266:nt+=1;twins+=w;tties+=int(w==1)
                self.assertAlmostEqual(exact[0],wins/1980,places=12,msg=cards)
                self.assertAlmostEqual(exact[1],twins/(2*nt),places=12,msg=cards)
                self.assertAlmostEqual(exact[2],ties/990,places=12,msg=cards)
                self.assertAlmostEqual(exact[3],tties/nt,places=12,msg=cards)

    def test_straight_flush_quads_and_blocked_flush_cards(self):
        boards=([0,4,8,12,16],[48,44,40,36,32],[0,4,8,12,1],
                [48,49,50,51,44],[0,1,4,5,8],[0,4,8,13,14])
        for board in boards:
            for hole in list(combinations(sorted(set(range(52))-set(board)),2))[::100]:
                cards=list(hole)+list(board);remain=sorted(set(range(52))-set(cards));score=evaluate(cards)
                total=twins=nt=0
                for opp in combinations(remain,2):
                    os=evaluate(opp+tuple(board));w=2 if score>os else 1 if score==os else 0
                    total+=w
                    if _PREFLOP['order'][hand_key(opp)]<266:nt+=1;twins+=w
                exact=exact_river(hole,board)
                self.assertAlmostEqual(exact[0],total/1980,places=12,msg=cards)
                self.assertAlmostEqual(exact[1],twins/(2*nt),places=12,msg=cards)

    def test_production_river_branch_skips_model(self):
        old=equity.USE_EXACT_RIVER
        try:
            equity.USE_EXACT_RIVER=True
            with patch.object(equity,'_infer',side_effect=AssertionError('River must skip the model')):
                for cards in ([48,49,0,4,8,12,16],[43,17,45,44,49,30,50],
                              [0,1,48,44,40,36,32]):
                    a=equity.analyze(cards[:2],cards[2:]);exact=equity.exact_river(cards[:2],cards[2:],with_ties=True)
                    self.assertTrue(a['exact_river'])
                    self.assertEqual((a['equity_random'],a['equity_tight']),exact[:2])
                    self.assertEqual((a['tie_random'],a['tie_tight']),exact[2:])
        finally:
            equity.USE_EXACT_RIVER=old

if __name__=='__main__':unittest.main()
