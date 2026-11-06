"""Independent checks of bot13's offline HU shove/fold game asset.

The tests check the learned two-action abstraction, not full NLHE equilibrium.
"""
from __future__ import annotations
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RANKS = "23456789TJQKA"


def key(a, b):
    hi, lo = max(a//4, b//4), min(a//4, b//4)
    return RANKS[hi]+RANKS[lo]+("" if hi == lo else "s" if a%4 == b%4 else "o")


def representative(name):
    return [4*RANKS.index(name[0]), 4*RANKS.index(name[1])+(0 if name.endswith("s") else 1)]


def independent_best_response(data, scenario):
    count = data["class_combo_counts"]
    combos = data["conditional_combo_counts"]
    eq = data["equity_matrix"]
    x, y, depth = scenario["shove_probs"], scenario["call_probs"], scenario["effective_stack_bb"]
    value = sb_best = baseline = 0.0
    coefficients = [0.0]*169
    for i in range(169):
        conditional_value = 0.0
        for j in range(169):
            called = eq[i][j]*2*depth-(depth-.5)
            conditional_value += combos[i][j]/1225*((1-y[j])*1.5+y[j]*called)
            joint = count[i]/1326*combos[i][j]/1225
            coefficients[j] += joint*x[i]*(called-1.5)
        value += count[i]/1326*x[i]*conditional_value
        sb_best += count[i]/1326*max(0, conditional_value)
        baseline += count[i]/1326*x[i]*1.5
    bb_best = baseline+sum(min(0, c) for c in coefficients)
    return value, sb_best, bb_best


@unittest.skipUnless((ROOT / "assets" / "pushfold.json").exists(), "generate optional offline pushfold asset first")
class PushFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "assets" / "pushfold.json").read_text(encoding="utf-8"))

    def test_exact_class_and_blocker_conditioned_chance_counts(self):
        data = self.data
        actual = Counter(key(a, b) for a, b in combinations(range(52), 2))
        self.assertEqual(len(data["classkeys"]), 169)
        self.assertEqual(len(set(data["classkeys"])), 169)
        self.assertEqual(sum(data["class_combo_counts"]), 1326)
        for i, name in enumerate(data["classkeys"]):
            self.assertEqual(data["class_combo_counts"][i], actual[name])
            hero = representative(name)
            available = [c for c in range(52) if c not in hero]
            counts = Counter(key(a, b) for a, b in combinations(available, 2))
            self.assertEqual(sum(data["conditional_combo_counts"][i]), 1225)
            self.assertEqual(data["conditional_combo_counts"][i], [counts[k] for k in data["classkeys"]])
            for j in range(169):
                self.assertEqual(data["conditional_combo_counts"][i][j]*data["class_combo_counts"][i],
                                 data["conditional_combo_counts"][j][i]*data["class_combo_counts"][j])

    def test_matrix_symmetry_diagonal_and_equal_stratified_runouts(self):
        data = self.data
        for i in range(169):
            self.assertEqual(data["equity_matrix"][i][i], .5)
            for j in range(169):
                q = data["equity_matrix"][i][j]
                self.assertTrue(math.isfinite(q) and 0 <= q <= 1)
                self.assertAlmostEqual(q+data["equity_matrix"][j][i], 1, places=8)
                if i < j:
                    n = data["actual_samples_per_cell"][i][j]
                    combos = data["conditional_combo_counts"][i][j]
                    self.assertEqual(n%combos, 0)
                    self.assertGreaterEqual(n, data["requested_samples_per_cell"])
                    self.assertLess(n, data["requested_samples_per_cell"]+combos)

    def test_chip_gain_formulas_conserve_and_match_stack_outcomes(self):
        for depth in (2, 3, 5, 10, 15):
            for q in (0, .2, .5, .8, 1):
                sb_gain = q*2*depth-(depth-.5)
                bb_gain = (1-q)*2*depth-(depth-1)
                self.assertAlmostEqual(sb_gain+bb_gain, 1.5)
                self.assertAlmostEqual(bb_gain, 1.5-sb_gain)
                self.assertAlmostEqual(sb_gain, q*(depth+.5)+(1-q)*(-depth+.5))
                self.assertAlmostEqual(bb_gain, (1-q)*(depth+1)+q*(-depth+1))

    def test_independent_exact_best_response_certificates(self):
        self.assertFalse(self.data["full_nlhe_equilibrium_claimed"])
        self.assertEqual([d["effective_stack_bb"] for d in self.data["depths"]], [2,3,4,5,6,7,8,10,12,15])
        for scenario in self.data["depths"]:
            for field in ("shove_probs", "call_probs"):
                self.assertEqual(len(scenario[field]), 169)
                self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in scenario[field]))
            value, sb_best, bb_best = independent_best_response(self.data, scenario)
            self.assertAlmostEqual(value, scenario["sb_value_vs_fold_bb"], places=7)
            self.assertAlmostEqual(value-.5, scenario["sb_net_value_bb"], places=7)
            self.assertAlmostEqual(sb_best, scenario["sb_best_response_value_bb"], places=7)
            self.assertAlmostEqual(bb_best, scenario["bb_best_response_sb_value_bb"], places=7)
            self.assertAlmostEqual(sb_best-bb_best, scenario["nash_conv_bb"], places=7)
            self.assertAlmostEqual((sb_best-bb_best)/2, scenario["exploitability_bb"], places=7)
            self.assertGreaterEqual(sb_best-value, -1e-7)
            self.assertGreaterEqual(value-bb_best, -1e-7)

    def test_selected_equities_against_independent_weighted_mc_sampler(self):
        from bot13.native import NativeEvaluator, native_available
        if not native_available:
            self.skipTest("optional native equity kernel unavailable")
        engine = NativeEvaluator()
        for hero_key, opp_key in (("AA", "KK"), ("AKs", "QQ"), ("72o", "AKo"), ("JJ", "T9s")):
            i, j = self.data["classkeys"].index(hero_key), self.data["classkeys"].index(opp_key)
            weights = [float(key(a,b) == opp_key) for a, b in combinations(range(52), 2)]
            q, error = engine.equity(representative(hero_key), [], 1, weights, seed=131561+i*169+j, samples=50000)
            tolerance = 5*math.sqrt(error*error+self.data["matrix_standard_errors"][i][j]**2)+.002
            self.assertAlmostEqual(q, self.data["equity_matrix"][i][j], delta=tolerance)

    def test_analytical_mixed_strategy_toy(self):
        executable = ROOT / "native" / ("b13_pushfold.exe" if sys.platform == "win32" else "b13_pushfold")
        if not executable.exists():
            self.skipTest("offline trainer binary not packaged")
        # The executable's 2-type independent-chance toy has analytical solution
        # x=(1,.2), y=(1,.8) and SB incremental value=.54 at depth2.
        subprocess.run([str(executable), "--self-test"], check=True, cwd=ROOT, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
