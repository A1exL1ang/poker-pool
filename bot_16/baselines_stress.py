"""Additional independent adversarial DEVELOPMENT opponents.

These are stress behaviors, not strong solved policies or contest submissions.
Kept separate from baselines.py so an in-flight frozen tuning run is unaffected.
"""
from random import Random
from baselines import Baseline
from simulator import evaluate


class StressBaseline:
    def __init__(self, style, seed=0):
        if style not in ('minraise', 'calling', 'polar'):
            raise ValueError(style)
        self.style = style
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = Random(seed)

    @staticmethod
    def raise_to(obs, amount):
        if not obs['can_raise']:
            return {'action': 'call' if obs['to_call'] else 'check'}
        return {'action': 'raise', 'amount': min(obs['max_raise_to'], max(obs['min_raise_to'], round(amount)))}

    def act(self, obs):
        passive = {'action': 'call' if obs['to_call'] else 'check'}
        fold = {'action': 'fold' if obs['to_call'] else 'check'}
        if self.style == 'minraise':
            return self.raise_to(obs, obs['min_raise_to'])
        hero = obs['players'][obs['hero']]
        call = min(obs['to_call'], hero['stack'])
        bb, pot = obs['big_blind'], obs['pot']
        ranks = sorted((c // 4 + 2 for c in obs['hole']), reverse=True)
        pair = ranks[0] == ranks[1]
        board = obs['board']
        if self.style == 'calling':
            if not board:
                strong = pair and ranks[0] >= 13
            else:
                rank = evaluate(obs['hole'] + board)
                strong = rank[0] >= 6 and (len(board) < 5 or rank != evaluate(board))
            if strong and obs['can_raise']:
                return self.raise_to(obs, hero['street_bet'] + call + max(bb, pot + call))
            return passive
        # Polar strategy: deliberately conspicuous large bets, with a small
        # bluff component. It tests whether size alone overwhelms range logic.
        opponents = sum(not p['folded'] for p in obs['players']) - 1
        bluff = self.rng.random() < (.035 if opponents == 1 else .012)
        if not board:
            premium = (pair and ranks[0] >= 12) or ranks == [14, 13]
            if obs['can_raise'] and (premium or bluff):
                target = max(8 * bb, 4 * (hero['street_bet'] + call),
                             hero['street_bet'] + call + 1.5 * (pot + call))
                return self.raise_to(obs, target)
            return passive if premium else fold
        rank = evaluate(obs['hole'] + board)
        category = rank[0]
        contributes = len(board) < 5 or rank != evaluate(board)
        strong = category >= 3 and contributes
        if obs['can_raise'] and (strong or bluff):
            return self.raise_to(obs, hero['street_bet'] + call + max(bb, 1.6 * (pot + call)))
        price = call / max(1, pot + call)
        # Strong hands continue, as do top pairs at modest pot odds; most other
        # hands give up facing bets instead of becoming another calling station.
        top_pair = category == 1 and rank[1] in ranks and rank[1] >= max(c // 4 + 2 for c in board)
        return passive if not call or strong or (top_pair and price < .28) else fold


STRESS_FIELDS = {
    'adversarial': ['minraise', 'calling', 'calling', 'polar', 'polar', 'minraise', 'maniac'],
    'calling': ['calling'] * 7,
    'minraise': ['minraise'] * 7,
    'polar': ['polar'] * 7,
}


def make_stress_field(name, seed=0):
    return [Baseline(style, seed + i) if style in Baseline.STYLES else StressBaseline(style, seed + i)
            for i, style in enumerate(STRESS_FIELDS[name])]
