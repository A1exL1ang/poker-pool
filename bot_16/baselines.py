"""Independent synthetic benchmark opponents (not contest entrants).

These intentionally cover different exploitable styles. They are heuristic
baselines, not game-theoretic equilibria or evidence of contest strength.
`equity` uses bounded Monte Carlo and may exceed the contest decision limit;
benchmark reports its timing separately and does not claim it is admissible.
"""
from random import Random
from simulator import evaluate


class Baseline:
    STYLES = ('tag', 'lag', 'nit', 'station', 'maniac', 'random', 'equity', 'jam', 'fold')

    def __init__(self, style='tag', seed=0):
        if style not in self.STYLES:
            raise ValueError(style)
        self.style = style
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = Random(seed)

    @staticmethod
    def _raise(obs, target):
        if not obs['can_raise']:
            return {'action': 'call' if obs['to_call'] else 'check'}
        target = min(obs['max_raise_to'], max(obs['min_raise_to'], round(target)))
        return {'action': 'raise', 'amount': target}

    def _equity(self, hole, board, opponents, samples=24):
        known = set(hole + board)
        deck = [c for c in range(52) if c not in known]
        won = 0.0
        for _ in range(samples):
            draw = self.rng.sample(deck, 5 - len(board) + 2 * opponents)
            runout = board + draw[:5 - len(board)]
            hero = evaluate(hole + runout)
            ties, beaten = 1, False
            for j in range(opponents):
                start = 5 - len(board) + 2 * j
                villain = evaluate(draw[start:start + 2] + runout)
                if villain > hero:
                    beaten = True
                    break
                ties += villain == hero
            if not beaten:
                won += 1.0 / ties
        return (won + .5) / (samples + 1)

    def act(self, obs):
        style, rng = self.style, self.rng
        hero = obs['players'][obs['hero']]
        call = min(obs['to_call'], hero['stack'])
        pot, bb = obs['pot'], obs['big_blind']
        price = call / max(1, pot + call)
        passive = {'action': 'call' if call else 'check'}
        fold = {'action': 'fold' if call else 'check'}
        if style == 'fold':
            return fold
        if style == 'jam':
            return self._raise(obs, obs['max_raise_to'])
        if style == 'random':
            u = rng.random()
            if u < .18 and obs['can_raise']:
                return self._raise(obs, hero['street_bet'] + call + max(bb, (pot + call) * rng.choice((.33, .66, 1))))
            return passive if not call or u < .65 else fold
        hole, board = obs['hole'], obs['board']
        active = sum(not p['folded'] for p in obs['players'])
        ranks = sorted((c // 4 + 2 for c in hole), reverse=True)
        high, low = ranks
        pair, suited = high == low, hole[0] % 4 == hole[1] % 4
        # Independent smooth preflop ordering; no assets from candidate bot.
        quality = .23 + .023 * high + .012 * low + .055 * suited - .012 * max(0, high - low - 1)
        if pair:
            quality = .48 + .026 * high
        quality = min(.91, quality)
        if not board:
            late = obs['hero'] == obs['button']
            raised = sum(a['action'] == 'raise' for a in obs['history'] if a['street'] == 0)
            thresholds = {'nit': .73, 'tag': .64, 'lag': .54, 'station': .43, 'maniac': .37, 'equity': .62}
            threshold = thresholds[style] - .045 * late - .10 * (active == 2)
            stackbb = (hero['stack'] + hero['street_bet']) / bb
            if style == 'station':
                if quality > .83 and obs['can_raise']:
                    return self._raise(obs, max(3 * bb, 3 * (hero['street_bet'] + call)))
                return passive if quality >= threshold and price < .45 else fold
            if stackbb <= 10 and quality > threshold:
                return self._raise(obs, obs['max_raise_to'])
            if raised:
                if quality > .82 - .055 * (style == 'maniac') and obs['can_raise']:
                    return self._raise(obs, 3.2 * (hero['street_bet'] + call))
                return passive if quality > threshold + .025 * raised and price < (.36 if style != 'nit' else .26) else fold
            if quality > threshold and obs['can_raise']:
                return self._raise(obs, (2.5 + .5 * (style == 'maniac')) * bb)
            return passive if call <= .5 * bb and quality > threshold - .09 else fold
        if style == 'equity':
            equity = self._equity(hole, board, active - 1)
        else:
            rank = evaluate(hole + board)
            category = rank[0]
            board_high = max(c // 4 + 2 for c in board)
            if category == 0:
                equity = .18 + .012 * high + .04 * (high > board_high)
            elif category == 1:
                pair_rank = rank[1]
                board_pairs = len({c // 4 for c in board}) < len(board)
                contributed = not board_pairs or pair_rank in ranks
                equity = .43 + .024 * pair_rank if contributed else .28 + .012 * high
                if contributed and pair_rank < board_high:
                    equity -= .10
            else:
                equity = {2: .81, 3: .89, 4: .94, 5: .96, 6: .985, 7: .998, 8: 1.0}[category]
                if len(board) == 5 and evaluate(board) == rank:
                    equity = .36
            equity = max(.03, min(.99, equity)) ** (max(1, active - 1) ** .65)
        # Draw encouragement for pre-river cards, with exact-card suit accounting.
        draw = False
        if len(board) < 5:
            suits = [sum(c % 4 == s for c in hole + board) for s in range(4)]
            draw = max(suits) == 4
        thresholds = {'nit': .76, 'tag': .68, 'lag': .57, 'station': .88, 'maniac': .40, 'equity': .68}
        value_threshold = thresholds[style]
        aggression = {'nit': .015, 'tag': .06, 'lag': .18, 'station': .005, 'maniac': .38, 'equity': .06}[style]
        u = rng.random()
        if obs['can_raise'] and (equity > value_threshold + .10 * bool(call) or (not call and u < aggression * (1.5 if draw else 1))):
            fraction = .70 if style in ('maniac', 'station') else .55
            target = hero['street_bet'] + call + max(bb, (pot + call) * fraction)
            return self._raise(obs, target)
        if not call:
            return passive
        margin = {'nit': .14, 'tag': .08, 'lag': .02, 'station': -.14, 'maniac': -.08, 'equity': .05}[style]
        if equity + (.08 if draw else 0) > price + margin:
            return passive
        return fold


FIELDS = {
    'mixed': ['tag', 'tag', 'lag', 'nit', 'station', 'maniac', 'equity'],
    'tight': ['nit', 'nit', 'tag', 'tag', 'tag', 'equity', 'equity'],
    'loose': ['station', 'station', 'lag', 'lag', 'maniac', 'maniac', 'random'],
    'aggressive': ['lag', 'lag', 'maniac', 'maniac', 'tag', 'tag', 'equity'],
    'tag': ['tag'] * 7,
    'equity': ['equity'] * 7,
    'jam': ['jam'] * 7,
    'fold': ['fold'] * 7,
}


def make_field(name='mixed', seed=0):
    return [Baseline(style, seed + i) for i, style in enumerate(FIELDS[name])]
