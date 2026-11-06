import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_pool20 import bootstrap_ratios,summary_stat,validate_row,COUNTERS
from schedule20 import iter_matches,VERSION

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.expected=next(iter_matches(1))
        self.row=dict(self.expected,version=VERSION,points=[.125]*8,
                      final_stacks=[2000]*8,placements=[4.5]*8,max_latency_ns=[100]*8)
        self.row.update({k:[0]*8 for k in COUNTERS})
        self.row["decisions"]=[1]*8
        self.row["total_latency_ns"]=[100]*8

    def test_valid_row_and_separate_budget_diagnostic(self):
        self.row["budget_fallbacks"][0]=1
        validate_row(self.row,self.expected)

    def test_rejects_schedule_drift(self):
        self.row["seats"]=self.row["seats"][1:]+self.row["seats"][:1]
        with self.assertRaisesRegex(ValueError,"schedule mismatch"):validate_row(self.row,self.expected)

    def test_rejects_fault_and_nonfinite_points(self):
        self.row["internal_errors"][0]=1
        with self.assertRaisesRegex(ValueError,"unexplained fault"):validate_row(self.row,self.expected)
        self.row["internal_errors"][0]=0;self.row["points"][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):validate_row(self.row,self.expected)

    def test_rejects_chip_loss(self):
        self.row["final_stacks"][0]-=1
        with self.assertRaisesRegex(ValueError,"chip conservation"):validate_row(self.row,self.expected)

class BootstrapTests(unittest.TestCase):
    def test_ratio_estimator_uses_exposure_counts(self):
        sums=np.array([[100.,-50.],[1.,-.5],[20.,-10.]])
        counts=np.array([[100.,100.],[1.,1.],[20.,20.]])
        x=bootstrap_ratios(sums,counts,64,batch_size=7)
        np.testing.assert_allclose(x[:,0],1.)
        np.testing.assert_allclose(x[:,1],-.5)
        self.assertEqual(summary_stat(sums[:,0],counts[:,0],x[:,0])["tables"],121)

    def test_clusters_are_resampled_whole(self):
        # Eight perfectly correlated seat rotations are one cluster, not8draws.
        x=bootstrap_ratios([[0.],[8.]],[[8.],[8.]],256)
        self.assertTrue(set(x[:,0])<={0.,.5,1.})
        np.testing.assert_allclose(np.quantile(x[:,0],[.025,.975]),[0.,1.])

    def test_reproducible_across_batch_size(self):
        a=bootstrap_ratios([[1.],[3.],[2.]],[[2.],[4.],[3.]],70,batch_size=7)
        b=bootstrap_ratios([[1.],[3.],[2.]],[[2.],[4.],[3.]],70,batch_size=32)
        np.testing.assert_array_equal(a,b)

if __name__=="__main__":unittest.main()
