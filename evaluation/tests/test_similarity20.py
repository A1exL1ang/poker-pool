"""Lightweight unit checks; runtime probes belong on Engaging."""
import ast
import tempfile
import unittest
from pathlib import Path

from similarity20 import Normalize, agreement_metrics, function_records, json_structure, is_legal
from common.engine import Action
from types import SimpleNamespace


class SimilarityMetricsTests(unittest.TestCase):
    def setUp(self):
        self.o=dict(to_call=20,stacks=[2000]*8,seat=0,pot=100,bb=20)

    def test_universal_folds_are_not_informative(self):
        result=agreement_metrics([('fold','fold',0,0,self.o)]*12)
        self.assertEqual(result['action_agreement'],1)
        self.assertEqual(result['both_fold'],1)
        self.assertEqual(result['nontrivial_count'],0)
        self.assertIsNone(result['chance_corrected_agreement'])

    def test_raise_sizing_and_nontrivial_denominators(self):
        result=agreement_metrics([('raise','raise',120,130,self.o),
                                  ('call','fold',0,0,self.o),
                                  ('fold','fold',0,0,self.o)])
        self.assertEqual(result['both_raise_count'],1)
        self.assertEqual(result['nontrivial_count'],2)
        self.assertAlmostEqual(result['nontrivial_agreement'],.5)
        self.assertAlmostEqual(result['mean_raise_difference_pot_units'],10/120)
        self.assertEqual(result['raises_within_5pct_pot_or_1bb'],1)

    def test_alpha_normalization_retains_constants(self):
        a=Normalize().visit(ast.parse('def f(x):\n y=x+2\n return y\n').body[0])
        b=Normalize().visit(ast.parse('def g(v):\n z=v+2\n return z\n').body[0])
        c=Normalize().visit(ast.parse('def h(v):\n z=v+3\n return z\n').body[0])
        self.assertEqual(ast.dump(a),ast.dump(b))
        self.assertNotEqual(ast.dump(a),ast.dump(c))

    def test_asset_shape_omits_values(self):
        result=json_structure({'trees':[[.123,.456],[.789,.1]],'version':4})
        self.assertEqual(result['children']['trees']['length'],2)
        self.assertNotIn('.123',str(result))

    def test_action_legality_short_raise_and_closed_rights(self):
        o=SimpleNamespace(to_call=20,street_bets=[0]*8,seat=0,can_raise=True,max_raise_to=30,min_raise_to=30)
        self.assertTrue(is_legal(Action('raise',30),o))
        self.assertFalse(is_legal(Action('raise',29),o))
        o.can_raise=False
        self.assertFalse(is_legal(Action('raise',30),o))
        self.assertTrue(is_legal(Action('call'),o))


if __name__=='__main__':unittest.main()
