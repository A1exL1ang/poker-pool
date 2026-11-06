"""Original local sparring opponents; these are not other teams' bots.

They deliberately span styles and are not evidence of equilibrium strength.
"""
import random
from poker_eval import features
from equity_model import preflop_percentile, preflop_equity, predict_from_features


class Baseline:
    def __init__(self, style='tag', seed=0):
        self.style, self.seed = style, seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)

    @staticmethod
    def raise_to(o, target):
        if 'raise' in o['legal_actions']:
            return {'action': 'raise', 'amount': min(o['max_raise_to'], max(o['min_raise_to'], int(target)))}
        return {'action': 'check' if not o['to_call'] else 'call', 'amount': 0}

    def act(self, o):
        h = o['hero']
        call = min(o['to_call'], o['stacks'][h])
        passive = {'action': 'check' if not call else 'fold', 'amount': 0}
        style = self.style
        if style == 'fold':
            return passive
        if style == 'station':
            return {'action': 'call' if call else 'check', 'amount': 0}
        if style == 'random':
            u = self.rng.random()
            if u < .14:
                return self.raise_to(o, max(o['bets']) + max(o['big_blind'], o['pot']*.55))
            if u < .63:
                return {'action': 'call' if call else 'check', 'amount': 0}
            return passive
        if style == 'maniac':
            if self.rng.random() < .55:
                return self.raise_to(o, max(o['bets']) + max(o['big_blind']*3, o['pot']*.9))
            return {'action': 'call' if call else 'check', 'amount': 0}
        villains = [i for i, f in enumerate(o['folded']) if not f and i != h]
        n = len(villains)
        cap = o['contributions'][h] + call
        pot = sum(min(c, cap) for c in o['contributions']) if call else o['pot']
        odds = call / max(1, pot+call)
        pct = preflop_percentile(o['hole'])
        loose = style == 'lag'
        tight = style == 'nit'
        if not o['board']:
            raised = max(o['bets']) > o['big_blind']
            if style == 'shover':
                depth = (o['stacks'][h] + o['bets'][h]) / o['big_blind']
                threshold = (.42 if depth < 8 else .25 if depth < 18 else .11) / (1 + .10*(n-1))
                if pct < threshold:
                    return self.raise_to(o, o['max_raise_to'])
                return passive
            if raised:
                width = .42 if loose else .12 if tight else .24
                eq = preflop_equity(o['hole'], width, max(1, sum(o['bets'][v] > o['big_blind'] for v in villains)))
                if pct < (.06 if not tight else .025):
                    return self.raise_to(o, max(o['bets'])*3)
                if pct < (.46 if loose else .16 if tight else .29) and eq*.85 > odds+.035:
                    return {'action': 'call', 'amount': 0}
                return passive
            threshold = .46 if loose else .13 if tight else .26
            if h == o['button']:
                threshold *= 1.5
            if n == 1:
                threshold *= 1.5
            if pct < threshold:
                return self.raise_to(o, o['big_blind']*(3 if loose else 2.5))
            return passive
        raises = sum(e['action'] == 'raise' and e['street'] != 'preflop' for e in o['history'])
        width = .35 if loose else .15 if tight else .28
        f = features(o['hole'], o['board'], width, min(.7, raises*.25))
        hu = predict_from_features(f)
        eq = hu ** (max(1, n)**.85)
        if call and eq < odds + (.09 if tight else .025):
            return passive
        value = hu > (.82 if tight else .64 if loose else .73) + .045*(n-1)
        bluff = not tight and n <= 2 and self.rng.random() < (.19 if loose else .065) and not call
        if value or bluff:
            target = max(o['bets']) + int(max(o['big_blind'], (pot+call)*(.85 if loose else .60)))
            if target > .65 * o['max_raise_to']:
                target = o['max_raise_to']
            return self.raise_to(o, target)
        return {'action': 'call' if call else 'check', 'amount': 0}


POOLS = {
    'mixed': ['nit', 'tag', 'station', 'lag', 'tag', 'random', 'shover'],
    'tight': ['nit', 'tag', 'nit', 'tag', 'tag', 'tag', 'nit'],
    'pressure': ['lag', 'maniac', 'tag', 'shover', 'lag', 'tag', 'maniac'],
    'calling': ['station', 'station', 'tag', 'random', 'station', 'lag', 'tag'],
    'mirror': ['mirror'] * 7,
    # Held-out-only stress pool; never included by benchmarks/tune.py.
    'diverse': ['agent_tight','agent_loose','agent_nobluff','agent_call',
                'agent_pressure','agent_nolearn','agent_safe'],
}


def make_baseline(name, seed=0):
    if name.startswith('agent_'):
        from agent import Agent
        params={'open_scale':1.0,'call_margin':.025,'value_threshold':.66,
                'bluff_scale':1.0,'range_tightness':1.0,'realization':.92,
                'push_margin':.08,'adapt':True,'icm':True}
        variants={
            'agent_tight':{'open_scale':.8,'call_margin':.06,'value_threshold':.72},
            'agent_loose':{'open_scale':1.4,'call_margin':0.,'range_tightness':1.3},
            'agent_nobluff':{'bluff_scale':0.},
            'agent_call':{'realization':.98,'call_margin':0.,'value_threshold':.75},
            'agent_pressure':{'bluff_scale':1.6,'value_threshold':.62,'open_scale':1.2},
            'agent_nolearn':{'adapt':False},
            'agent_safe':{'bluff_scale':.25,'call_margin':.05,'open_scale':.95},
        }
        params.update(variants[name])
        return Agent(seed=seed,params=params)
    return Baseline(name, seed)
