import itertools
from pathlib import Path
import random
import sys
import subprocess
import unittest

from pokerbot.equity import FEATURE_NAMES, feature_vector, hand_features, has_absolute_nuts
from pokerbot.evaluator import card_id, evaluate


def cards(text):
    return [card_id(card) for card in text.split()]


class EquityFeaturesTest(unittest.TestCase):
    def test_schema(self):
        vector = feature_vector(cards("As Ks"), cards("Qs Js 2h"))
        self.assertEqual(len(vector), len(FEATURE_NAMES))
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))
        features = dict(zip(FEATURE_NAMES, vector))
        self.assertEqual(features["total_max_suit"], 4)
        self.assertEqual(features["flush_nut_gap"], 0)
        self.assertEqual(features["straight_out_ranks"], 1)
        self.assertEqual(features["hole_overcards"], 2)

    def test_pair_suit_invariance(self):
        for hole_text,board_text in (("As Ad","Js 8s 2d 3h 5c"),("7s 7d","2s 3s Ad Kd 8c")):
            hole, board = cards(hole_text), cards(board_text)
            expected = feature_vector(hole, board)
            self.assertEqual(expected, feature_vector(hole[::-1], board))
            for perm in itertools.permutations(range(4)):
                convert = lambda card: card // 4 * 4 + perm[card % 4]
                self.assertEqual(feature_vector([convert(c) for c in hole], [convert(c) for c in board]), expected)

    def test_suit_and_hole_order_invariance(self):
        rng = random.Random(14988)
        for _ in range(150):
            n = rng.choice([3, 4, 5])
            sample = rng.sample(range(52), n + 2)
            hole, board = sample[:2], sample[2:]
            expected = feature_vector(hole, board)
            self.assertEqual(expected, feature_vector(hole[::-1], board[::-1]))
            perm = rng.sample(range(4), 4)
            transformed = [c // 4 * 4 + perm[c % 4] for c in sample]
            self.assertEqual(feature_vector(transformed[:2], transformed[2:]), expected)

    def test_nuts_detector_is_conservative(self):
        self.assertTrue(has_absolute_nuts(cards("As Ks"), cards("Qs Js Ts")))
        self.assertTrue(has_absolute_nuts(cards("Ac Ad"), cards("Ah As 7d 4c 2h")))
        self.assertFalse(has_absolute_nuts(cards("2c 2d"), cards("2h 2s Kc Kd Kh")))
        # Quads lose to a possible straight flush when its suit has three
        # relevant board cards. The conservative detector returns false.
        self.assertFalse(has_absolute_nuts(cards("Ac Ad"), cards("Ah As Kh Qh Jh")))
        self.assertTrue(has_absolute_nuts(cards("Ac Kd"), cards("2c 2d 2h 2s 4c")))
        self.assertFalse(has_absolute_nuts(cards("Kc Qd"), cards("2c 2d 2h 2s 4c")))
        self.assertTrue(has_absolute_nuts(cards("2d 3s"), cards("Ac Ad Ah As Kc")))
        self.assertTrue(has_absolute_nuts(cards("Kh 3s"), cards("Ac Ad Ah As Qc")))
        self.assertFalse(has_absolute_nuts(cards("Qh 3s"), cards("Ac Ad Ah As Jc")))
        self.assertTrue(has_absolute_nuts(cards("Ac Kd"), cards("Qh Js Tc 4d 2h")))
        self.assertFalse(has_absolute_nuts(cards("Ac Kd"), cards("Qh Jh Th 4d 2c")))
        self.assertFalse(has_absolute_nuts(cards("Ac Kd"), cards("Qh Qs Tc Jd 2h")))

    def test_cpp_python_parity_if_compiled(self):
        root = Path(__file__).resolve().parents[1]
        candidates = [root / ("tools/equity_assets.exe" if sys.platform == "win32" else "tools/equity_assets")]
        executable = next((path for path in candidates if path.exists()), None)
        if executable is None:
            self.skipTest("optional C++ probe has not been compiled")
        rng = random.Random(140002)
        hands = []
        for _ in range(1000):
            n = rng.choice([3, 4, 5])
            cards = rng.sample(range(52), n + 2)
            hands.append(cards)
        payload = "\n".join(" ".join(map(str, [len(hand) - 2, *hand])) for hand in hands) + "\n"
        completed = subprocess.run([str(executable), "--probe"], input=payload, text=True, capture_output=True, check=True, timeout=20)
        rows = completed.stdout.splitlines()
        self.assertEqual(len(rows), len(hands))
        for row, hand in zip(rows, hands):
            actual = tuple(map(int, row.split()))
            expected = (evaluate(hand), *feature_vector(hand[:2], hand[2:]))
            self.assertEqual(actual, expected, hand)


if __name__ == "__main__":
    unittest.main()
