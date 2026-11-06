import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('latency',ROOT/'tools/latency.py')
LATENCY=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LATENCY)


class LatencyTests(unittest.TestCase):
    def test_quantiles_preserve_slow_calls(self):
        values=[10,20,30,1_000_001]
        result=LATENCY.distribution(values)
        self.assertEqual(result['n'],4)
        self.assertEqual(result['over_1ms'],1)
        self.assertEqual(result['max_ns'],1_000_001)
        self.assertEqual(result['mean_ns'],sum(values)/4)

    def test_stateful_replay_matches_reachable_original_actions(self):
        for lineup in ('mixed','mirror'):
            tables,collection=LATENCY.collect(lineup,100,seed_start=9890,max_tables=4,max_hands=100)
            self.assertEqual(collection['recorded_decisions'],100)
            result=LATENCY.replay(tables)
            self.assertEqual(result['all']['wall']['n'],100)
            self.assertEqual(result['all']['process']['n'],100)
            self.assertEqual(result['action_mismatches'],0)
            self.assertEqual(result['invalid_actions'],0)
            self.assertEqual(result['uncaught_exceptions'],0)
            self.assertEqual(result['internal_safety_fallbacks'],0)
            self.assertEqual(result['preflop']['wall']['n']+result['postflop']['wall']['n'],100)


if __name__=='__main__':
    unittest.main()
