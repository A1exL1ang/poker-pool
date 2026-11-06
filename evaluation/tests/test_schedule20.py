import sys
import unittest
from collections import Counter
from itertools import combinations
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from schedule20 import design,fields_for,iter_matches,keyed_int,metadata
from run_pool20 import SeedSeparated

class ScheduleTests(unittest.TestCase):
    def test_exact_field_incidence_and_pair_balance(self):
        for cohort,b,k,r,lam in (("pool",285,8,114,42),("mixed",570,6,171,45)):
            f=design(cohort)
            self.assertEqual(len(f),b)
            self.assertEqual(Counter(x for row in f for x in row),Counter({x:r for x in range(20)}))
            self.assertEqual(Counter(pair for row in f for pair in combinations(row,2)),Counter({p:lam for p in combinations(range(20),2)}))

    def test_seat_rotations_and_baseline_exposure(self):
        counts=Counter();base=Counter();ids=set();seeds={}
        for row in iter_matches(1,payouts=("top3",)):
            self.assertNotIn(row["match_id"],ids);ids.add(row["match_id"])
            self.assertEqual(len(set(row["seats"])),8)
            for seat,n in enumerate(row["seats"]):counts[row["cohort"],n,seat]+=1
            if row["cohort"]=="mixed":
                self.assertTrue({21,22}<=set(row["seats"]))
                for n in row["seats"]:
                    if n<=20:base[n]+=1
            seeds.setdefault(row["cluster_id"],set()).add(row["seed"])
        self.assertEqual(len(ids),6840)
        for c,r in (("pool",114),("mixed",171)):
            for n in range(1,21):
                for seat in range(8):self.assertEqual(counts[c,n,seat],r)
        self.assertEqual(set(base.values()),{1368})
        self.assertTrue(all(len(x)==1 for x in seeds.values()))
        self.assertEqual(len({next(iter(x)) for x in seeds.values()}),855)

    def test_production_count_and_repeatability(self):
        self.assertEqual(metadata(4)["tables"],82080)
        self.assertEqual(fields_for("pool",2),fields_for("pool",2))
        self.assertNotEqual(fields_for("pool",1),fields_for("pool",2))
        self.assertNotEqual(keyed_int("deck","x"),keyed_int("policy","x"))

    def test_engine_seed_never_reaches_adapter(self):
        class Fake:
            table_id=None
            def reset(self,*args):self.received=args
        a=Fake();w=SeedSeparated(a,3,"pool:0:4")
        w.reset(987654321,2,8)
        self.assertEqual(a.received,(keyed_int("policy","pool:0:4",3),2,8))
        self.assertNotEqual(a.received[0],987654321)
        self.assertTrue(a.table_id.startswith("table-"))
        first=a.table_id
        w.reset(111111,2,8)
        self.assertEqual(a.table_id,first)

if __name__=="__main__":unittest.main()
