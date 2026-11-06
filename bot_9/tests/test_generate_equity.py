"""Offline-generator tests; skipped when a local C++17 compiler is unavailable."""
import csv
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('g++'), 'C++17 compiler unavailable; runtime bot does not require it')
class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        control = ROOT / '.control'
        control.mkdir(exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(prefix='generator-test-', dir=control)
        cls.path = Path(cls.directory.name).resolve()
        assert cls.path.is_relative_to(ROOT)
        cls.binary = cls.path / 'generator.exe'
        subprocess.run([shutil.which('g++'), '-O2', '-std=c++17',
                        str(ROOT/'tools/generate_equity.cpp'), '-o', str(cls.binary)],
                       cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)

    @classmethod
    def tearDownClass(cls):
        assert cls.path.is_relative_to(ROOT)
        cls.directory.cleanup()

    def run_generator(self, *args):
        return subprocess.run([str(self.binary), *map(str, args)], cwd=ROOT,
                              check=True, capture_output=True, text=True, timeout=30)

    def test_exact_made_prior_mechanics(self):
        result = self.run_generator('validate-made')
        self.assertIn('royal board ties', result.stdout)
        self.assertIn('whole boundary ties', result.stdout)
        self.assertIn('suit invariance', result.stdout)

    def test_samples6_schema_determinism_and_original_compatibility(self):
        old, new, repeat = (self.path/name for name in ('old.csv', 'new.csv', 'repeat.csv'))
        common = [12, 64, 719, ROOT/'assets/preflop.csv']
        self.run_generator('samples', old, *common)
        result = self.run_generator('samples6', new, *common)
        self.run_generator('samples6', repeat, *common)
        self.assertEqual(new.read_bytes(), repeat.read_bytes())
        self.assertIn('Whole-tie current-made actual mass bounds', result.stderr)
        with old.open(newline='') as f:
            old_rows = list(csv.DictReader(f))
        with new.open(newline='') as f:
            new_rows = list(csv.DictReader(f))
        self.assertEqual(len(old_rows), 12)
        self.assertEqual(len(new_rows), 12)
        self.assertEqual(len(old_rows[0]), 11)
        self.assertEqual(len(new_rows[0]), 13)
        self.assertEqual(set(new_rows[0])-set(old_rows[0]), {'eq_made40', 'eq_made15'})
        # First row's original labels/cards consume the exact same RNG stream.
        self.assertEqual(old_rows[0], {k: new_rows[0][k] for k in old_rows[0]})
        for index, row in enumerate(new_rows):
            card_ids = [int(row[k]) for k in ('h0', 'h1', 'b0', 'b1', 'b2', 'b3', 'b4')]
            known = [card for card in card_ids if card >= 0]
            self.assertEqual(len(known), 5+index % 3)
            self.assertEqual(len(set(known)), len(known))
            self.assertTrue(all(0 <= card < 52 for card in known))
            self.assertTrue(all(0 <= float(value) <= 1 for k, value in row.items() if k.startswith('eq_')))

    def test_invalid_rollout_count_rejected(self):
        result = subprocess.run([str(self.binary), 'samples6', str(self.path/'bad.csv'),
                                 '1', '0', '719', str(ROOT/'assets/preflop.csv')],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('positive rollouts', result.stderr)


if __name__ == '__main__':
    unittest.main()
