import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from bot3.interface import validate_observation, validate_action


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.o = json.loads(Path("examples/observation.json").read_text())

    def test_example_and_jsonlines_process(self):
        validate_observation(self.o)
        inp = json.dumps(self.o) + '\n{"command":"reset","table_id":"second"}\n' + json.dumps(self.o) + "\n"
        r = subprocess.run([sys.executable, "-m", "bot3"], input=inp, capture_output=True, text=True, check=True)
        results = [json.loads(x) for x in r.stdout.splitlines()]
        self.assertEqual(results[0], {"action": "raise", "amount": 48})
        self.assertEqual(results[1], {"ok": True})
        self.assertEqual(results[0], results[2])

    def test_duplicate_cards_missing_fields_and_units(self):
        for change in ({"hole_cards": ["As", "As"]}, {"to_call": 19}, {"pot": 30.5}, {"payouts": [1, 2]}, {"board": [0]}):
            o = copy.deepcopy(self.o)
            o.update(change)
            with self.assertRaises(ValueError):
                validate_observation(o)
        o = copy.deepcopy(self.o)
        del o["hero_seat"]
        with self.assertRaises(ValueError):
            validate_observation(o)

    def test_below_minimum_all_in_only(self):
        o = copy.deepcopy(self.o)
        o["players"][2]["stack"] = 30
        o["max_raise_to"] = 30
        validate_action(o, {"action": "raise", "amount": 30})
        with self.assertRaises(ValueError):
            validate_action(o, {"action": "raise", "amount": 29})
