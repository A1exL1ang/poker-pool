"""Independent, deliberately diverse local opponents; not contest opponents."""
import random

from benchmarks.engine import evaluate


def preflop_strength(hole):
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    suited = hole[0] % 4 == hole[1] % 4
    if a == b:
        return min(.99, .57 + a * .029)
    return min(.94, .12 + a * .034 + b * .017 + .055 * suited
               + .035 * (a - b <= 2) - .02 * (a - b >= 6))


def strength(obs):
    hole, board = obs['hole'], obs['board']
    if not board:
        return preflop_strength(hole)
    rank = evaluate(hole + board)
    cat = rank[0]
    values = [.20, .50, .72, .82, .90, .94, .98, .995, 1.0]
    value = values[cat]
    top = max(c // 4 + 2 for c in board)
    ranks = [c // 4 + 2 for c in hole]
    if cat == 1:
        pair = rank[1]
        if pair not in ranks:
            value = .24 + max(ranks) / 140
        else:
            value += .13 * (pair >= top) + .03 * (max(ranks) >= 12)
    if len(board) == 5 and evaluate(board) == rank:
        value = min(value, .30)
    if len(board) < 5:
        suits = [sum(c % 4 == s for c in hole + board) for s in range(4)]
        if max(suits) == 4 and any(sum(c % 4 == s for c in hole) > 0 and suits[s] == 4 for s in range(4)):
            value = max(value, .42 if len(board) == 3 else .30)
    return value


class BaselineAgent:
    def __init__(self, style='balanced', seed=0):
        self.style = style
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)

    def act(self, obs):
        hero = obs['hero']
        stack = obs['stacks'][hero]
        call = min(stack, obs['to_call'])
        check = {'action': 'check' if call == 0 else 'fold', 'amount': 0}
        calling = {'action': 'call' if call else 'check', 'amount': call}
        if self.style == 'fold':
            return check

        def raise_to(amount):
            if not obs['can_raise']:
                return calling
            target = min(obs['max_raise_to'], max(obs['min_raise_to'], int(amount)))
            return {'action': 'raise', 'amount': target}

        s = strength(obs)
        pot = obs['pot']
        odds = call / max(1, pot + call)
        bb = obs['big_blind']
        bet = obs['street_bets'][hero]
        if self.style == 'random':
            r = self.rng.random()
            if r < .20:
                return check
            if r < .80:
                return calling
            return raise_to(bet + call + max(bb, pot * .5))
        if self.style == 'shove':
            if not obs['street'] and s > (.67 if stack > bb * 15 else .47):
                return raise_to(obs['max_raise_to'])
            if call and s > (.72 if obs['street'] == 0 else odds + .34):
                return calling
            return check
        if self.style == 'calling':
            if call and (not obs['street'] and s < .41 and call > bb * 6
                         or obs['street'] and s < odds * .85):
                return check
            if s > .93 and self.rng.random() < .55:
                return raise_to(bet + call + max(bb, pot * .7))
            return calling
        tight = self.style == 'tight'
        aggressive = self.style == 'aggressive'
        if obs['street'] == 0:
            unopened = max(obs['street_bets']) <= bb
            position = obs.get('position', 3)
            threshold = (.69 if tight else .59 if aggressive else .63) + .008 * position
            if unopened and s >= threshold:
                return raise_to(bb * (3 if aggressive else 2.5))
            if s >= (.88 if tight else .83):
                return raise_to(bet + call + max(3 * bb, pot * .85))
            if call and s >= threshold - .06 and call <= max(2 * bb, stack * .09):
                return calling
            if call and s > .78 and odds < .29:
                return calling
            return check
        if call:
            if s > (.86 if tight else .80):
                return raise_to(bet + call + max(bb, pot * .7))
            threshold = odds + (.29 if tight else .16 if aggressive else .22)
            if s > threshold:
                return calling
            if aggressive and self.rng.random() < .04 and call < stack * .25:
                return raise_to(bet + call + pot)
            return check
        if s > (.66 if tight else .56 if aggressive else .61):
            return raise_to(bet + max(bb, pot * (.8 if aggressive else .55)))
        if self.rng.random() < (.19 if aggressive else .035 if tight else .09):
            return raise_to(bet + max(bb, pot * .55))
        return check
