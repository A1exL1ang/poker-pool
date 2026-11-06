import random
import subprocess
import sys
import unittest
from pathlib import Path

from bot3.cards import card_id, card_name, evaluate, features, reference_evaluate, hand_class


def cards(s):
    return [card_id(c) for c in s.split()]


class CardTests(unittest.TestCase):
    def test_encoding_roundtrip(self):
        for i in range(52):
            self.assertEqual(card_id(card_name(i)), i)
        self.assertEqual(card_id("10S"), card_id("Ts"))
        for bad in (-1, 52, "Ax", "1s", True):
            with self.assertRaises(ValueError):
                card_id(bad)

    def test_all_categories_and_kickers(self):
        hands = ["As Kd Qh 9c 7s 4d 2h", "As Ad Qh 9c 7s 4d 2h", "As Ad Qh Qc 7s 4d 2h",
                 "As Ad Ah Qc 7s 4d 2h", "As 2d 3h 4c 5s Kd Qh", "As Js 9s 7s 4s 2d Qh",
                 "As Ad Ah Qc Qs 4d 2h", "As Ad Ah Ac Qs 4d 2h", "As Ks Qs Js Ts 4d 2h"]
        values = [evaluate(cards(h)) for h in hands]
        self.assertEqual([v >> 20 for v in values], list(range(9)))
        self.assertEqual(values, sorted(values))
        self.assertGreater(evaluate(cards("As Ad Ah Ks Kd Kh 2s")), evaluate(cards("Qs Qd Qh As Ad 3c 2s")))
        self.assertEqual(evaluate(cards("As Ks Qs Js Ts 2d 3c")), evaluate(cards("As Ks Qs Js Ts 8d 9c")))

    def test_independent_oracle(self):
        rng = random.Random(781)
        for n in (5, 6, 7):
            for _ in range(4000):
                cs = rng.sample(range(52), n)
                self.assertEqual(evaluate(cs), reference_evaluate(cs), cs)

    def test_suit_invariance(self):
        rng = random.Random(19)
        for _ in range(300):
            cs = rng.sample(range(52), 7)
            perm = rng.sample(range(4), 4)
            ds = [(c & ~3) | perm[c & 3] for c in cs]
            self.assertEqual(evaluate(cs), evaluate(ds))
            self.assertEqual(features(cs[:2], cs[2:]), features(ds[:2], ds[2:]))
            self.assertEqual(hand_class(*cs[:2]), hand_class(*ds[:2]))

    def test_native_feature_parity(self):
        exe = Path("tools/offline.exe" if sys.platform == "win32" else "tools/offline")
        if not exe.exists():
            self.skipTest("optional native offline executable absent")
        rng = random.Random(683)
        cases = []
        for _ in range(1500):
            n = rng.choice((3, 4, 5))
            cs = rng.sample(range(52), n + 2)
            cases.append((n, cs))
        inp = "".join(" ".join(map(str, [n, *cs])) + "\n" for n, cs in cases)
        result = subprocess.run([str(exe.resolve()), "features"], input=inp, capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        self.assertEqual(len(lines), len(cases))
        for (_, cs), line in zip(cases, lines):
            self.assertEqual(features(cs[:2], cs[2:]), list(map(int, line.split())), cs)
