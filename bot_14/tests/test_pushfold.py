import contextlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from tools.train_pushfold import audit, class_names, conditional_counts, toy_validation, train


class PushFoldSemanticsTest(unittest.TestCase):
    def test_exact_card_blockers(self):
        counts, combos = conditional_counts()
        names = class_names()
        self.assertEqual(sum(combos), 1326)
        self.assertEqual({sum(row) for row in counts}, {1225})
        aa, aks, ako, twotwo = (names.index(name) for name in ("AA", "AKs", "AKo", "22"))
        self.assertEqual(counts[aa][aa], 1)
        self.assertEqual(counts[aa][aks], 2)
        self.assertEqual(counts[aa][ako], 6)
        self.assertEqual(counts[aa][twotwo], 6)
        self.assertEqual(counts[aks][aa], 3)
        self.assertEqual(counts[aks][aks], 3)
        self.assertEqual(counts[ako][ako], 7)
        for i, j in itertools.product(range(169), repeat=2):
            self.assertEqual(combos[i] * counts[i][j], combos[j] * counts[j][i])

    def test_equal_equity_toy(self):
        for report in toy_validation():
            self.assertEqual(report["sb_shove"], 1)
            self.assertEqual(report["bb_call"], 1)
            self.assertEqual(report["gap_bb"], 0)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is an offline-training dependency")
    def test_audit_against_explicit_two_type_game(self):
        import numpy as np

        equity = np.array([[.5, .8], [.2, .5]])
        prior = np.array([.4, .6])
        conditional = np.array([[.25, .75], [.5, .5]])
        depth = 6
        a = (conditional * ((2 * equity - 1) * depth - 1))[None, :, :]
        b = (conditional * ((2 * equity - 1) * depth + 1))[None, :, :]
        shove, call = np.array([.3, .7]), np.array([.2, .9])

        def value(s, c):
            result = 0.0
            for i, j in itertools.product(range(2), repeat=2):
                called = (2 * equity[i, j] - 1) * depth
                result += prior[i] * conditional[i, j] * ((1 - s[i]) * -.5 + s[i] * ((1 - c[j]) + c[j] * called))
            return result

        report = audit(np, prior, conditional, a, b, shove[None, :], call[None, :])[0]
        current = value(shove, call)
        sb_br = max(value(policy, call) for policy in itertools.product((0, 1), repeat=2))
        bb_br = min(value(shove, policy) for policy in itertools.product((0, 1), repeat=2))
        self.assertAlmostEqual(report["sb_value_bb"], current)
        self.assertAlmostEqual(report["sb_best_response_gain_bb"], sb_br - current)
        self.assertAlmostEqual(report["bb_best_response_gain_bb"], current - bb_br)
        self.assertAlmostEqual(report["nash_conv_gap_bb"], sb_br - bb_br)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is an offline-training dependency")
    def test_training_equal_equity_asset(self):
        counts, combos = conditional_counts()
        asset = {"classes": class_names(), "equity": [[.5] * 169 for _ in range(169)]}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(depths=[1.0, 10.0], iterations=30, average_delay=0,
                                   checkpoint_every=10, checkpoint=str(Path(directory) / "checkpoint.json"),
                                   output=str(Path(directory) / "pushfold.json"), resume=None)
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(args, asset, counts, combos, "synthetic-equal-equity")
            self.assertEqual(result["sb_shove"], [[1.0] * 169] * 2)
            self.assertEqual(result["bb_call"], [[1.0] * 169] * 2)
            self.assertLess(max(row["nash_conv_gap_bb"] for row in result["audit"]), 1e-12)
            self.assertTrue(Path(args.checkpoint).exists())
            self.assertEqual(json.loads(Path(args.output).read_text())["iterations"], 30)


if __name__ == "__main__":
    unittest.main()
