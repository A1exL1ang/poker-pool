"""Offline generator integration tests; compile its binary before running.

g++ -O3 -std=c++17 -fopenmp tools/generate_equity.cpp -o tools/generate_equity
python -m unittest discover -s tests -p test_generator.py -v
"""

import json
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

from pokerbot.evaluator import evaluate


ROOT = Path(__file__).resolve().parents[1]
BINARY = next((p for p in (ROOT / "tools/generate_equity.exe", ROOT / "tools/generate_equity") if p.exists()), None)


@unittest.skipIf(BINARY is None, "compile tools/generate_equity.cpp to enable integration tests")
class GeneratorTests(unittest.TestCase):
    def run_generator(self, *args, input=None):
        return subprocess.run([str(BINARY), *map(str, args)], input=input, text=True,
                              cwd=ROOT, capture_output=True, check=True, timeout=90)

    def test_cpp_python_scores_match(self):
        rng = random.Random(11021)
        hands = [rng.sample(range(52), size) for size in (5, 6, 7) for _ in range(500)]
        result = self.run_generator("--mode", "evaluate", input="\n".join(" ".join(map(str, hand)) for hand in hands))
        self.assertEqual(list(map(int, result.stdout.split())), [evaluate(hand) for hand in hands])

    def test_record_layout_ranges_and_thread_determinism(self):
        with tempfile.TemporaryDirectory(prefix="bot_11_equity_test_") as temp:
            single = Path(temp) / "single.bin"
            parallel = Path(temp) / "parallel.bin"
            arguments = ("--mode", "postflop", "--states", 60, "--rollouts", 256,
                         "--ranking-rollouts", 1000, "--seed", 11022)
            self.run_generator(*arguments, "--threads", 1, "--out", single)
            self.run_generator(*arguments, "--threads", 2, "--out", parallel)
            data = single.read_bytes()
            self.assertEqual(data, parallel.read_bytes())
            self.assertEqual(len(data), 60 * 48)
            streets = {3: 0, 4: 0, 5: 0}
            for record in struct.iter_unpack("<8B10f", data):
                boardlen, *values = record
                cards, labels = values[:7], values[7:]
                self.assertIn(boardlen, streets)
                streets[boardlen] += 1
                self.assertEqual(len(set(cards[:boardlen + 2])), boardlen + 2)
                self.assertTrue(all(0 <= card < 52 for card in cards[:boardlen + 2]))
                self.assertTrue(all(card == 255 for card in cards[boardlen + 2:]))
                self.assertTrue(all(0 <= value <= 1 for value in labels))
                self.assertGreaterEqual(labels[0] + 1e-7, labels[1])
                self.assertGreaterEqual(labels[1] + 1e-7, labels[2])
            self.assertEqual(streets, {3: 20, 4: 20, 5: 20})

    def test_preflop_json_and_fundamental_equities(self):
        with tempfile.TemporaryDirectory(prefix="bot_11_preflop_test_") as temp:
            output = Path(temp) / "preflop.json"
            self.run_generator("--mode", "preflop", "--rollouts", 3000,
                               "--ranking-rollouts", 1000, "--seed", 11023,
                               "--threads", 2, "--out", output)
            result = json.loads(output.read_text())
            self.assertEqual(len(result), 169)
            self.assertEqual(sum(len(key) == 2 for key in result), 13)
            for key, row in result.items():
                self.assertIn(len(key), (2, 3))
                self.assertTrue(0 <= row["percentile"] <= 1)
                self.assertEqual(len(row["equity"]), 7)
                self.assertTrue(all(0 <= equity <= 1 for equity in row["equity"]))
                self.assertGreaterEqual(row["equity"][0] + 1e-7, row["uniform2"])
                self.assertGreaterEqual(row["uniform2"] + 1e-7, row["uniform3"])
            self.assertGreater(result["AA"]["equity"][0], .80)
            self.assertLess(result["72o"]["equity"][0], .40)
            self.assertLess(result["AA"]["percentile"], result["72o"]["percentile"])


if __name__ == "__main__":
    unittest.main()
