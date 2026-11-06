import itertools
import random
import unittest

from bot.evaluator import card
from bot.features import hand_index, make_features, VERSION
from bot import equity


class EquityTests(unittest.TestCase):
    def test_canonical_hand_indices_exhaustive(self):
        counts = [0] * 169
        for a in range(52):
            for b in range(a + 1, 52):
                i = hand_index([a, b])
                self.assertEqual(i, hand_index([b, a]))
                counts[i] += 1
        self.assertEqual(counts, [6] * 13 + [4] * 78 + [12] * 78)

    def test_suit_order_and_board_order_invariance(self):
        rng = random.Random(161602)
        for nb in (3, 4, 5):
            for i in range(100):
                cards = rng.sample(range(52), nb + 2)
                f = make_features(cards[:2], cards[2:], 2, .25, 1)
                self.assertEqual(len(f), 58 if VERSION == 2 else 84)
                p = list(range(4))
                rng.shuffle(p)
                z = [c // 4 * 4 + p[c % 4] for c in cards]
                self.assertEqual(f, make_features(z[:2][::-1], z[2:][::-1], 2, .25, 1))

    def test_royal_is_unbeatable(self):
        hole = [card('As'), card('Ks')]
        board = [card(s) for s in ('Qs', 'Js', 'Ts')]
        self.assertEqual(equity.postflop(hole, board, .10, 7, 2), 1.)

    def test_royal_board_is_chopped(self):
        hole = [card('2c'), card('3h')]
        board = [card(s) for s in ('As', 'Ks', 'Qs', 'Js', 'Ts')]
        for n in range(1, 8):
            self.assertEqual(equity.postflop(hole, board, .1, n, 2), 1. / (n + 1))

    @unittest.skipUnless(equity.MODEL_AVAILABLE, 'model not generated')
    def test_board_only_pot_share_upper_bound(self):
        h = [card('2c'), card('3d')]
        for b in [('Ac', 'Ad', 'Ah', 'Ks', 'Kd'),
                  ('7c', '7d', '7h', '7s', 'As'),
                  ('8c', '9d', 'Th', 'Js', 'Qc')]:
            board = [card(c) for c in b]
            for n in range(1, 8):
                self.assertLessEqual(equity.postflop(h, board, 1., n, 0), 1. / (n + 1))

    @unittest.skipUnless(equity.COMPILED_AVAILABLE, 'compiled asset not generated')
    def test_compiled_reference_exact_parity(self):
        rng = random.Random(161117)
        for _ in range(300):
            cards = rng.sample(range(52), 7)
            f = make_features(cards[:2], cards[2:], rng.randrange(1, 8), rng.random(), rng.random() * 2)
            self.assertEqual(equity.predict(f), equity.predict_reference(f))

    @unittest.skipUnless(equity.MODEL_AVAILABLE and equity.PREFLOP_AVAILABLE, 'assets not yet generated')
    def test_assets_and_broad_preflop_order(self):
        aa = [card('As'), card('Ah')]
        qq = [card('Qs'), card('Qh')]
        bad = [card('7s'), card('2h')]
        self.assertLess(equity.percentile(aa), equity.percentile(qq))
        self.assertLess(equity.percentile(qq), equity.percentile(bad))
        self.assertGreater(equity.preflop(aa), .82)
        self.assertLess(equity.preflop(bad), .38)
        self.assertGreater(equity.preflop(aa, opponents=1), equity.preflop(aa, opponents=7))
        for h in (aa, qq, bad):
            for n in range(1, 8):
                for r in (.01, .1, .2, .25, .4, .5, 1, 1.5):
                    self.assertTrue(0 <= equity.preflop(h, r, n) <= 1)

    @unittest.skipUnless(equity.MODEL_AVAILABLE, 'model not yet generated')
    def test_feature_reuse_preserves_inputs(self):
        hole = [card('As'), card('Qh')]
        board = [card(s) for s in ('Ac', '7h', '2s')]
        features = make_features(hole, board)
        copy = list(features)
        expected = equity.postflop(hole, board, .25, 3, 2)
        got = equity.postflop(hole, board, .25, 3, 2, features=features)
        self.assertEqual(expected, got)
        self.assertEqual(features, copy)


if __name__ == '__main__':
    unittest.main()
