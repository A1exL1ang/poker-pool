"""Optional C++ generator integration tests (skipped without a local binary)."""
import csv
import io
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = next((p for p in (ROOT / 'tools/generate_equity.exe', ROOT / 'tools/generate_equity') if p.exists()), None)


@unittest.skipUnless(BINARY, 'Build the optional C++ generator first')
class GeneratorTests(unittest.TestCase):
    def test_samples_reproducible_legal_bounded(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            paths = [pathlib.Path(directory) / f'sample{i}.csv' for i in range(2)]
            for path in paths:
                subprocess.run([str(BINARY), 'samples', str(path), '60', '128', '41871', str(ROOT / 'assets/preflop.csv')], check=True, capture_output=True)
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())
            rows = list(csv.DictReader(io.StringIO(paths[0].read_text())))
            self.assertEqual(len(rows), 60)
            for i, row in enumerate(rows):
                cards = [int(row[key]) for key in ('h0', 'h1', 'b0', 'b1', 'b2', 'b3', 'b4') if int(row[key]) >= 0]
                self.assertEqual(len(cards), len(set(cards)))
                self.assertEqual(len(cards), 5 + i % 3)
                self.assertTrue(all(0 <= c < 52 for c in cards))
                self.assertTrue(all(0 <= float(row[k]) <= 1 for k in ('eq_random', 'eq_top40', 'eq_top15', 'eq_top5')))

    def test_independent_cpp_evaluator(self):
        result = subprocess.run([str(BINARY), 'validate', '3000', '99185'], check=True, capture_output=True, text=True)
        self.assertIn('Validated 3000', result.stdout)


if __name__ == '__main__':
    unittest.main()
