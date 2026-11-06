"""Independent checks of the finite first-caller push/fold abstraction."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("solve_pushfold", ROOT / "tools/solve_pushfold.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
action_values = MODULE.action_values


class PushFoldTests(unittest.TestCase):
    def test_uncalled_pot_and_blind_accounting(self):
        for depth in (2, 8, 15):
            for behind in range(1, 8):
                opener, callers = action_values([[.5]], [[1.]], [1.], [[0.]] * behind, depth)
                self.assertEqual(opener, [1.5])
                hero_cost = depth - (.5 if behind == 1 else 0)
                for j, values in enumerate(callers):
                    cost = depth - (1 if j == behind - 1 else .5 if j == behind - 2 else 0)
                    self.assertAlmostEqual(values[0], .5 * (1.5 + hero_cost + cost) - cost)

    def test_first_caller_blocks_later_callers(self):
        for behind in range(1, 8):
            opener, callers = action_values([[.5]], [[1.]], [1.], [[1.]] * behind, 8)
            self.assertAlmostEqual(opener[0] + callers[0][0], 1.5)
            self.assertTrue(all(values == [0.] for values in callers[1:]))

    def test_folded_opener_never_reaches_callers(self):
        _, callers = action_values([[.5]], [[1.]], [0.], [[.5]] * 7, 10)
        self.assertEqual(callers, [[0.]] * 7)

    def test_asset_structure_and_diagnostics(self):
        path = ROOT / "assets/pushfold.json"
        if not path.exists():
            self.skipTest("generate assets/pushfold.json first")
        asset = json.loads(path.read_text())
        self.assertEqual(len(asset["keys"]), 169)
        self.assertEqual(len(set(asset["keys"])), 169)
        self.assertEqual(len(asset["diagnostics"]), 56)
        for depth in (2, 3, 4, 6, 8, 10, 12, 15):
            for behind in range(1, 8):
                shove = asset["shove"][str(depth)][str(behind)]
                call = asset["call"][str(depth)][str(behind)]
                self.assertEqual(len(shove), 169)
                self.assertEqual(len(call), behind)
                self.assertTrue(all(len(row) == 169 for row in call))
                self.assertTrue(all(0 <= x <= 1 for x in shove + sum(call, [])))
        for diag in asset["diagnostics"]:
            self.assertLess(diag["chip_accounting_error"], 1e-9)
            self.assertTrue(all(x >= 0 for x in diag["regret_bb_per_dealt_hand"]))

    def test_chip_identity_and_stored_regret_independently(self):
        path = ROOT / "assets/pushfold.json"
        if not path.exists():
            self.skipTest("generate assets/pushfold.json first")
        asset = json.loads(path.read_text())
        matchups = json.loads((ROOT / "assets/preflop_matchups.json").read_text())
        self.assertEqual(asset["keys"], matchups["keys"])
        p = [count / 1326 for count in matchups["combo_counts"]]
        cond = [[count / 1225 for count in row] for row in matchups["legal_opponent_counts"]]
        for depth, behind in ((2, 1), (8, 3), (15, 7)):
            shove = asset["shove"][str(depth)][str(behind)]
            calls = asset["call"][str(depth)][str(behind)]
            ov, cv = action_values(matchups["equity"], cond, shove, calls, depth)
            payoff = sum(w * s * v for w, s, v in zip(p, shove, ov))
            payoff += sum(sum(w * c * v for w, c, v in zip(p, strat, vals)) for strat, vals in zip(calls, cv))
            self.assertAlmostEqual(payoff, 1.5 * sum(w * s for w, s in zip(p, shove)), places=10)
            regret = [sum(w * (max(v, 0) - s * v) for w, s, v in zip(p, shove, ov))]
            regret += [sum(w * (max(v, 0) - c * v) for w, c, v in zip(p, strat, vals)) for strat, vals in zip(calls, cv)]
            diag = next(d for d in asset["diagnostics"] if d["depth_bb"] == depth and d["behind"] == behind)
            for actual, expected in zip(regret, diag["regret_bb_per_dealt_hand"]):
                self.assertAlmostEqual(actual, expected, places=6)


if __name__ == "__main__":
    unittest.main()
