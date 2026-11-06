"""Agent contract, invariance, and economically critical sanity checks."""
import copy
import random
import unittest

from bot import Agent, _icm_hero
from engine import Hand, icm
from evaluator import evaluate
from features import equity_features


class FixedModel:
    def __init__(self, value):
        self.value = value

    def predict(self, hole, board):
        return (self.value,) * 4


def spot(n=2):
    return dict(hero=0, hole=[48, 49], board=[0, 5, 10, 15, 20],
                stacks=[1000] * n, committed=[0] * n, contributions=[0] * n,
                folded=[False] * n, all_in=[False] * n, button=n-1,
                street=3, pot=0, to_call=0, min_raise_to=20,
                max_raise_to=1000, can_raise=True, bb=20, hand_no=0,
                payouts=None, history=[])


def rotate(observation, shift):
    o = copy.deepcopy(observation)
    n = len(o['stacks'])
    for key in ('stacks', 'committed', 'contributions', 'folded', 'all_in', 'starting_stacks'):
        if key in o:
            old = o[key]
            o[key] = [old[(i-shift) % n] for i in range(n)]
    o['hero'] = (o['hero'] + shift) % n
    o['button'] = (o['button'] + shift) % n
    for item in o['history']:
        item['seat'] = (item['seat'] + shift) % n
    return o


class BotTests(unittest.TestCase):
    def test_hero_icm_matches_independent_recursive_reference(self):
        rng = random.Random(779)
        for n in range(2, 6):
            for _ in range(20):
                stacks = [rng.randrange(1, 4000) for _ in range(n)]
                payouts = sorted([rng.randrange(100) for _ in range(n)], reverse=True)
                expected = icm(stacks, payouts)
                for hero in range(n):
                    self.assertAlmostEqual(_icm_hero(stacks, payouts, hero), expected[hero], places=9)

    def test_feature_suit_and_card_order_invariance(self):
        rng = random.Random(881)
        for n in (3, 4, 5):
            for _ in range(100):
                cards = rng.sample(range(52), n + 2)
                hole, board = cards[:2], cards[2:]
                permutation = rng.sample(range(4), 4)
                remapped = [c // 4 * 4 + permutation[c % 4] for c in cards]
                self.assertEqual(equity_features(hole, board), equity_features(remapped[:2], remapped[2:]))
                self.assertEqual(equity_features(hole, board), equity_features(hole[::-1], board[::-1]))

    def test_reachable_actions_legal_seat_and_suit_invariant(self):
        rng = random.Random(98190)
        agent, twin = Agent(seed=94), Agent(seed=94)
        decisions = 0
        for game in range(100):
            n = 8
            live = rng.sample(range(n), rng.randrange(2, 9))
            stacks = [rng.randrange(1, 4000) if i in live else 0 for i in range(n)]
            # Each independent table starts with cleared public opponent state.
            agent._hand, agent._history_index, agent._seen_pre, agent.stats = None, 0, set(), []
            hand = Hand(stacks, live[game % len(live)], 10, 20, rng=rng, evaluator=evaluate)
            while not hand.done:
                obs = hand.observation()
                shift = rng.randrange(n)
                twin.state = agent.state
                twin._hand = agent._hand
                twin._history_index = agent._history_index
                twin._seen_pre = {(i+shift) % n for i in agent._seen_pre}
                twin.stats = [copy.copy(agent.stats[(i-shift) % n]) for i in range(n)] if agent.stats else []
                shifted = rotate(obs, shift)
                action = agent.act(obs)
                self.assertEqual(action, twin.act(shifted))
                # Engine rejects out-of-range raises and incorrect check/call semantics.
                hand.step(action)
                decisions += 1
        self.assertGreater(decisions, 500)
        self.assertEqual(agent.errors, 0)
        self.assertEqual(twin.errors, 0)

    def test_raise_to_clamp_and_no_reopen(self):
        o = spot()
        o.update(min_raise_to=120, max_raise_to=110, can_raise=True)
        self.assertEqual(Agent._raise(o, 200), {'action': 'raise', 'amount': 110})
        o.update(can_raise=False, to_call=30)
        self.assertEqual(Agent._raise(o, 200), {'action': 'call'})
        o['to_call'] = 0
        self.assertEqual(Agent._raise(o, 200), {'action': 'check'})

    def test_side_pot_not_available_to_short_stack(self):
        o = spot(3)
        o.update(hole=[40, 44], stacks=[100, 0, 0], committed=[0, 1000, 1000],
                 contributions=[0, 1000, 1000], all_in=[False, True, True],
                 pot=2000, to_call=100, can_raise=False, max_raise_to=100,
                 history=[dict(seat=1, street=3, action='raise', amount=1000),
                          dict(seat=2, street=3, action='call', amount=1000)])
        agent = Agent()
        agent.model = FixedModel(.55)
        self.assertEqual(agent.act(o), {'action': 'fold'})

    def test_guaranteed_chop_royal_board_cannot_fold(self):
        o = spot()
        o.update(hole=[1, 6], board=[32, 36, 40, 44, 48],
                 stacks=[1000, 0], committed=[0, 1000], contributions=[1, 1001],
                 all_in=[False, True], pot=1002, to_call=1000,
                 can_raise=False, history=[dict(seat=1, street=3, action='raise', amount=1000)])
        agent = Agent()
        agent.model = FixedModel(.5)
        self.assertEqual(agent.act(o), {'action': 'call'})

    def test_premium_unopened_raises_and_trash_folds(self):
        hand = Hand([2000] * 8, 0, 10, 20, rng=random.Random(9), evaluator=evaluate)
        o = hand.observation()
        o['hole'] = [48, 49]
        self.assertEqual(Agent().act(o)['action'], 'raise')
        o['hole'] = [0, 21]  # 72o, first to act at a full table.
        self.assertEqual(Agent().act(o)['action'], 'fold')


if __name__ == '__main__':
    unittest.main()
