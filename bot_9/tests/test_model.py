"""Small synthetic binary/JSON parity tests; no large duplicate model loads."""
import gzip
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest
from model import EquityModel

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('compact_model',ROOT/'tools/compact_model.py')
COMPACT=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPACT)


class ModelTests(unittest.TestCase):
    def test_binary_json_parity_four_and_six_outputs(self):
        rng=random.Random(6191)
        with tempfile.TemporaryDirectory(prefix='bot9-model-test-',dir=ROOT) as directory:
            for outputs in (4,6):
                def leaf(offset):
                    return [-1,[round((i+offset)/(outputs+6),7) for i in range(outputs)]]
                tree=[[0,3.5,1,4],[1,6.5,2,3],leaf(1),leaf(2),leaf(3)]
                # Near an integer boundary deliberately exercises float32 packing.
                boundary=[[1,11.9999999,1,2],leaf(2),leaf(4)]
                data={'metadata':{'trees':2,'targets':['target'+str(i) for i in range(outputs)]},'trees':[tree,boundary]}
                source=Path(directory)/f'model{outputs}.json.gz'
                target=Path(directory)/f'model{outputs}.bin'
                with gzip.open(source,'wt',encoding='utf-8') as stream:
                    json.dump(data,stream)
                COMPACT.export(source,target,2)
                original=EquityModel(source)
                compact=EquityModel(target)
                self.assertEqual(compact.outputs,outputs)
                self.assertTrue(compact.binary)
                for n in (3,4,5):
                    for _ in range(100):
                        cards=rng.sample(range(52),n+2)
                        expected=original.predict(cards[:2],cards[2:])
                        actual=compact.predict(cards[:2],cards[2:])
                        self.assertEqual(len(actual),outputs)
                        for a,b in zip(actual,expected):
                            self.assertAlmostEqual(a,b,delta=1e-7)


if __name__=='__main__':
    unittest.main()
