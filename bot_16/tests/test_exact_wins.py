import random
import unittest
from bot.evaluator import evaluate, card
from training.features_v3 import make_features
from bot.exact import guaranteed_river_win


def cards(s):
    return [card(x) for x in s.split()]


class ExactWinsTests(unittest.TestCase):
    def test_known_wins(self):
        for h, b in [('As Ah', 'Ac Ad Ks 2d 3s'),
                     ('As Js', '2s 7s 9s Kc Td'),
                     ('9s 8s', '7s 6s 5s Kc Td')]:
            self.assertTrue(guaranteed_river_win(make_features(cards(h), cards(b))), (h, b))

    def test_shared_nuts_and_non_nuts_are_not_wins(self):
        for h, b in [('2c 3d', 'As Kd Qh Jc Ts'),
                     ('2c 3d', '7c 7d 7h 7s As'),
                     ('8s 6s', '2s 7s 9s Kc Td'),
                     ('4s 3s', '7s 6s 5s Kc Td'),
                     ('2c 3d', 'As Ks Qs Js Ts'),
                     ('As Kh', 'Ac Ad Kc 7h 2s')]:
            self.assertFalse(guaranteed_river_win(make_features(cards(h), cards(b))), (h, b))

    def test_future_board_prevents_general_guarantee(self):
        self.assertFalse(guaranteed_river_win(make_features(cards('As Ah'), cards('Ac Ad Ks'))))

    def test_guarantees_against_exhaustive_opponents(self):
        rng = random.Random(161118)
        checked = 0
        for _ in range(100000):
            known = rng.sample(range(52), 7)
            if evaluate(known) >> 20 < 3:
                continue
            h, b = known[:2], known[2:]
            f = make_features(h, b)
            if not guaranteed_river_win(f):
                continue
            unseen = [c for c in range(52) if c not in known]
            score = evaluate(known)
            for i, a in enumerate(unseen):
                for z in unseen[i + 1:]:
                    self.assertGreater(score, evaluate(b + [a, z]), (h, b, a, z))
            checked += 1
            if checked == 120:
                break
        self.assertGreaterEqual(checked, 120)


if __name__ == '__main__':
    unittest.main()
