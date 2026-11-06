import unittest
from tools.simulate import Table, FixturePolicy, settle_pots
from bot6.agent import Agent


class SimulatorTest(unittest.TestCase):
    def test_sidepots(self):
        self.assertEqual(settle_pots([100, 300, 500], [False] * 3, [30, 20, 10]), [300, 400, 200])
        self.assertEqual(settle_pots([100, 300, 500], [False] * 3, [10, 20, 30]), [0, 0, 900])

    def test_folded_dead_money_and_odd_chip(self):
        self.assertEqual(settle_pots([101, 101, 101], [False, False, True], [50, 50, 0], button=0), [151, 152, 0])

    def test_heads_up_action_order(self):
        t = Table([FixturePolicy("calling", 1), FixturePolicy("calling", 2)], seed=0, capture=True)
        t.play_hand(20)
        first = t.observations[0]
        self.assertEqual(first["hero"], 0)
        self.assertEqual(first["button"], 0)
        flop = next(o for o in t.observations if len(o["board"]) == 3)
        self.assertEqual(flop["hero"], 1)

    def test_short_blind_does_not_force_uncontested_call(self):
        t = Table([FixturePolicy("fold", i) for i in range(4)], seed=2, stacks=[2, 1, 3000, 12000])
        t.play_hand(40)
        self.assertEqual(sum(t.stacks), 15003)

    def test_uncalled_refund(self):
        t = Table([FixturePolicy("calling", 1), FixturePolicy("calling", 2)])
        t.bets, t.contributions, t.stacks = [200, 70], [300, 170], [1700, 1830]
        t._refund_uncalled()
        self.assertEqual(t.bets, [70, 70])
        self.assertEqual(t.contributions, [170, 170])
        self.assertEqual(t.stacks, [1830, 1830])

    def test_short_raise_does_not_reopen(self):
        class Script:
            def __init__(self, actions):
                self.actions, self.seen = iter(actions), []
            def act(self, o):
                self.seen.append(o)
                return next(self.actions)
        a = Script([{"action": "raise", "amount": 100}, {"action": "call"}])
        b = Script([{"action": "raise", "amount": 150}])
        c = Script([{"action": "call"}])
        t = Table([a, b, c], stacks=[1000, 150, 1000])
        t.live, t.bets, t.contributions, t.folded = [0, 1, 2], [0, 0, 0], [0, 0, 0], [False] * 3
        t.board, t.history, t.holes = [], [], {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        t._betting_round(0, 20, "flop")
        self.assertNotIn("raise", a.seen[-1]["legal_actions"])
        self.assertEqual(t.contributions, [150, 150, 150])

    def test_cumulative_short_raises_reopen(self):
        class Script:
            def __init__(self, actions):
                self.actions, self.seen = iter(actions), []
            def act(self, o):
                self.seen.append(o)
                return next(self.actions)
        a = Script([{"action": "raise", "amount": 100}, {"action": "call"}])
        b = Script([{"action": "raise", "amount": 150}])
        c = Script([{"action": "raise", "amount": 200}])
        d = Script([{"action": "call"}])
        t = Table([a, b, c, d], stacks=[1000, 150, 200, 1000])
        t.live, t.bets, t.contributions, t.folded = list(range(4)), [0] * 4, [0] * 4, [False] * 4
        t.board, t.history, t.holes = [], [], {i: [i * 2, i * 2 + 1] for i in range(4)}
        t._betting_round(0, 20, "flop")
        self.assertIn("raise", a.seen[-1]["legal_actions"])
        self.assertEqual(t.contributions, [200, 150, 200, 200])

    def test_full_tournaments_conserve_chips(self):
        for seed in range(12):
            policies = [Agent(seed)] + [FixturePolicy("random", seed * 9 + i) for i in range(1, 8)]
            t = Table(policies, seed, max_hands=200)
            t.run()
            self.assertEqual(sum(t.stacks), 16000)
            self.assertGreaterEqual(min(t.stacks), 0)


if __name__ == "__main__":
    unittest.main()
