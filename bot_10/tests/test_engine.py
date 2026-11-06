import random
import unittest
from engine import Hand, play_hand
from opponents import CheckCallBot, FoldBot, ProfileBot


def deck_prefix(prefix):
    return prefix + [c for c in range(52) if c not in prefix]


class EngineTests(unittest.TestCase):
    def test_heads_up_blinds_and_order(self):
        seen = []
        def a(obs):
            seen.append((obs['street'], obs['hero'], obs['to_call']))
            return {'action': 'call' if obs['to_call'] else 'check'}
        result = play_hand([a, a], [2000, 2000], button=0, seed=1)
        self.assertEqual(seen[:2], [(0, 0, 10), (0, 1, 0)])
        self.assertEqual(next(x for x in seen if x[0] == 1)[1], 1)
        self.assertEqual(sum(result['stacks']), 4000)
        self.assertFalse(result['errors'])

    def test_multiway_order(self):
        seen = []
        def a(obs):
            seen.append((obs['street'], obs['hero']))
            return {'action': 'call' if obs['to_call'] else 'check'}
        result = play_hand([a] * 5, [100] * 5, button=0)
        self.assertEqual(seen[:5], [(0, 3), (0, 4), (0, 0), (0, 1), (0, 2)])
        self.assertEqual(next(x for x in seen if x[0] == 1)[1], 1)
        self.assertFalse(result['errors'])

    def test_uncontested(self):
        result = play_hand([FoldBot()] * 3, [100] * 3, button=0)
        self.assertEqual(result['stacks'], [100, 90, 110])
        self.assertEqual(result['board'], [])

    def test_short_blind_call_is_full_bringin(self):
        h = Hand([CheckCallBot()] * 3, [100, 100, 7], button=0)
        self.assertEqual(h.observation(0)['to_call'], 20)
        self.assertEqual(h.observation(0)['min_raise_to'], 40)

    def test_nominal_bringin_not_forced_into_dry_sidepot(self):
        r = play_hand([ProfileBot('random', 54080), ProfileBot('maniac', 54081),
                       ProfileBot('balanced', 54082), ProfileBot('maniac', 54083),
                       ProfileBot('loose', 54084)], [261, 191, 48, 121, 3919],
                      button=3, small_blind=400, big_blind=800, seed=5408)
        self.assertFalse(r['errors'])
        self.assertEqual(sum(r['stacks']), 4540)
        self.assertFalse(any(a['seat'] == 4 and a['action'] == 'fold' for a in r['history']))

    def test_raise_to_semantics(self):
        h = Hand([CheckCallBot()] * 3, [1000] * 3, button=0)
        h.apply_action(0, dict(action='raise', amount=60))
        self.assertEqual(h.players[0]['stack'], 940)
        self.assertEqual(h.observation(1)['to_call'], 50)
        self.assertEqual(h.observation(1)['min_raise_to'], 100)
        h.apply_action(1, dict(action='call'))
        self.assertEqual(h.players[1]['stack'], 940)
        with self.assertRaises(ValueError):
            h.apply_action(2, dict(action='raise', amount=90))

    def test_short_raise_does_not_reopen(self):
        h = Hand([CheckCallBot()] * 3, [1000, 70, 1000], button=0)
        h.apply_action(0, dict(action='raise', amount=60))
        h.apply_action(1, dict(action='raise', amount=70))
        self.assertFalse(h.observation(0)['can_raise'])
        self.assertTrue(h.observation(2)['can_raise'])
        self.assertEqual(h.observation(0)['to_call'], 10)
        with self.assertRaises(ValueError):
            h.apply_action(0, dict(action='raise', amount=110))

    def test_cumulative_short_raises_reopen(self):
        h = Hand([CheckCallBot()] * 4, [1000, 1000, 70, 100], button=1)
        h.apply_action(0, dict(action='raise', amount=60))
        h.apply_action(1, dict(action='call'))
        h.apply_action(2, dict(action='raise', amount=70))
        h.apply_action(3, dict(action='raise', amount=100))
        self.assertTrue(h.observation(0)['can_raise'])
        self.assertEqual(h.observation(0)['min_raise_to'], 140)

    def test_sidepots_and_unmatched_refund(self):
        # Assign known complete board and holdings directly to isolate settlement.
        h = Hand([CheckCallBot()] * 3, [50, 100, 200], button=0)
        h.players = [dict(seat=i, stack=0, bet=s, committed=s, folded=False, all_in=True)
                     for i, s in enumerate([50, 100, 200])]
        h.holes = [[48, 49], [44, 45], [40, 41]]  # AA > KK > QQ
        h.board = [0, 5, 10, 15, 24]
        h.settle()
        self.assertEqual(h.payouts, [150, 100, 100])

    def test_folded_money_and_tied_odd_chip(self):
        h = Hand([CheckCallBot()] * 3, [5, 5, 5], button=0, small_blind=1, big_blind=2)
        h.players = [dict(seat=i, stack=0, bet=5, committed=5, folded=(i == 0), all_in=True) for i in range(3)]
        h.holes = [[4, 8], [12, 16], [20, 24]]
        h.board = [32, 36, 40, 44, 48]  # board royal flush, first seat left of button gets odd chip
        h.settle()
        self.assertEqual(h.payouts, [0, 8, 7])

    def test_short_call_observation_and_eligible_pot(self):
        h = Hand([CheckCallBot()] * 2, [500, 100], button=0)
        h.apply_action(0, dict(action='raise', amount=500))
        o = h.observation(1)
        self.assertEqual((o['pot'], o['to_call']), (520, 80))
        self.assertFalse(o['can_raise'])
        cap = h.players[1]['committed'] + o['to_call']
        eligible_after_call = sum(min(p['committed'], cap) for p in h.players) + o['to_call']
        self.assertEqual(eligible_after_call, 200)
        self.assertNotEqual(o['pot'] + o['to_call'], eligible_after_call)
        h.apply_action(1, dict(action='call'))
        h.refund_uncalled()
        h.holes = [[0, 1], [48, 49]]
        h.board = [4, 9, 14, 23, 28]
        h.settle()
        self.assertEqual([p['stack'] for p in h.players], [400, 200])

    def test_no_betting_dry_sidepot(self):
        h = Hand([CheckCallBot()] * 2, [1000, 20], button=0)
        self.assertFalse(h.observation(0)['can_raise'])
        r = h.run()
        self.assertFalse(r['errors'])
        self.assertEqual(sum(r['stacks']), 1020)

    def test_invalid_action_fallback(self):
        r = play_hand([lambda obs: dict(action='bogus'), FoldBot()], [100, 100])
        self.assertEqual(len(r['errors']), 1)
        self.assertEqual(r['stacks'], [90, 110])

    def test_seeded_ten_thousand_hand_fuzz(self):
        profiles = ['tight', 'balanced', 'loose', 'station', 'maniac', 'random', 'nit']
        rng = random.Random(914432)
        for seed in range(10000):
            n = rng.randint(2, 8)
            stacks = [rng.randrange(0, 4001) for _ in range(n)]
            stacks[0], stacks[1] = rng.randrange(1, 4001), rng.randrange(1, 4001)
            bb = rng.choice([2, 20, 50, 100, 200, 400, 800, 1600])
            bots = [ProfileBot(rng.choice(profiles), seed * 10 + i) for i in range(n)]
            r = play_hand(bots, stacks, button=seed % n, small_blind=bb // 2, big_blind=bb, seed=seed)
            self.assertEqual(sum(r['stacks']), sum(stacks), seed)
            self.assertGreaterEqual(min(r['stacks']), 0, seed)
            self.assertFalse(r['errors'], (seed, r['errors']))

    def test_many_profiles_conserve_and_terminate(self):
        profiles = ['tight', 'balanced', 'loose', 'station', 'maniac', 'random', 'nit', 'balanced']
        for seed in range(100):
            bots = [ProfileBot(profile, seed * 10 + i) for i, profile in enumerate(profiles)]
            stacks = [20 + ((seed * 91 + i * 67) % 1999) for i in range(8)]
            r = play_hand(bots, stacks, button=seed % 8, seed=seed)
            self.assertEqual(sum(r['stacks']), sum(stacks))
            self.assertFalse(r['errors'], (seed, r['errors']))
            self.assertTrue(all(s >= 0 for s in r['stacks']))


if __name__ == '__main__':
    unittest.main()
