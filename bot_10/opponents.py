"""Diverse independent benchmark policies; these are synthetic opponents, not solvers."""
import random


def preflop_score(hole):
    r = sorted((c // 4 + 2 for c in hole), reverse=True)
    suited = hole[0] % 4 == hole[1] % 4
    if r[0] == r[1]:
        return min(.99, .48 + .038 * r[0])
    gap = r[0] - r[1]
    return max(.05, min(.95, .025 * (r[0] + r[1]) + .06 * suited + .035 * (gap == 1) - .025 * max(0, gap - 3)))


def made_score(hole, board):
    """Transparent coarse strength model, intentionally separate from candidate."""
    cards = hole + board
    counts = [0] * 13
    suits = [0] * 4
    for c in cards:
        counts[c // 4] += 1
        suits[c % 4] += 1
    if max(suits) >= 5:
        return .96
    mask = sum(1 << r for r, n in enumerate(counts) if n)
    straight = any((mask >> low) & 31 == 31 for low in range(9)) or mask & 4111 == 4111
    if max(counts) >= 4:
        return .995
    trips = [r for r, n in enumerate(counts) if n >= 3]
    pairs = [r for r, n in enumerate(counts) if n >= 2]
    if trips and len(pairs) >= 2:
        return .98
    if straight:
        return .91
    if trips:
        return .84
    hero_ranks = {c // 4 for c in hole}
    board_ranks = [c // 4 for c in board]
    if len(pairs) >= 2:
        return .78 if any(r in hero_ranks for r in pairs) else .38
    if pairs:
        rank = max(pairs)
        if rank in hero_ranks and any(c // 4 == rank for c in hole):
            higher = sum(r > rank for r in set(board_ranks))
            return (.73 if higher == 0 else .57 if higher == 1 else .42)
        return .25 + .02 * max(hero_ranks)
    # Four-card draws get some continuation value before river.
    draw = len(board) < 5 and (max(suits) == 4 or any(((mask >> low) & 31).bit_count() >= 4 for low in range(9)))
    return .24 + .015 * max(hero_ranks) + .10 * draw


class ProfileBot:
    """Profiles: tight, balanced, loose, station, maniac, random, nit."""
    def __init__(self, profile='balanced', seed=0):
        self.profile, self.rng = profile, random.Random(seed)

    def act(self, obs):
        profile, rng = self.profile, self.rng
        p = obs['players'][obs['hero']]
        bb, pot, call = obs['big_blind'], max(obs['pot'], 1), obs['to_call']
        strength = preflop_score(obs['hole']) if not obs['board'] else made_score(obs['hole'], obs['board'])
        odds = call / (pot + call) if call else 0
        if profile == 'random':
            if obs['can_raise'] and rng.random() < .25:
                target = min(obs['max_raise_to'], max(obs['min_raise_to'], p['bet'] + call + int(pot * rng.uniform(.3, 1.4))))
                return dict(action='raise', amount=target)
            return dict(action='call' if call and rng.random() < .65 else 'fold' if call else 'check')
        if profile == 'station':
            if call and strength < odds * .63 and call > 3 * bb:
                return dict(action='fold')
            if strength > .9 and obs['can_raise']:
                target = min(obs['max_raise_to'], max(obs['min_raise_to'], p['bet'] + call + int(.7 * pot)))
                return dict(action='raise', amount=target)
            return dict(action='call' if call else 'check')
        looseness = {'nit': -.17, 'tight': -.08, 'balanced': 0., 'loose': .09, 'maniac': .19}[profile]
        if not obs['board']:
            required = (.51 if call <= bb else .57 + min(.21, call / max(p['stack'] + call, 1) * .4)) - looseness
            if call and strength < required:
                return dict(action='fold')
            raise_threshold = .64 - looseness
            if obs['can_raise'] and (strength > raise_threshold or profile == 'maniac' and rng.random() < .30):
                target = max(obs['min_raise_to'], (3 * bb if call <= bb else p['bet'] + call + max(int(pot * .8), 2 * call)))
                return dict(action='raise', amount=min(obs['max_raise_to'], target))
            return dict(action='call' if call else 'check')
        required = max(.28, odds + .22 - looseness)
        if call and strength < required:
            return dict(action='fold')
        if obs['can_raise'] and (strength > .69 - looseness or not call and rng.random() < .16 + looseness):
            target = max(obs['min_raise_to'], p['bet'] + call + int(pot * (.8 if profile == 'maniac' else .6)))
            return dict(action='raise', amount=min(obs['max_raise_to'], target))
        return dict(action='call' if call else 'check')


class CheckCallBot:
    def act(self, obs):
        return dict(action='call' if obs['to_call'] else 'check')


class FoldBot:
    def act(self, obs):
        return dict(action='fold' if obs['to_call'] else 'check')
