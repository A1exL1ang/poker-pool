"""bot_10: independent offline no-limit Texas Hold'em policy.

The mapping interface is local, documented in README, not an official adapter.
All decision work is bounded; expensive calibration is done offline.
"""
from __future__ import annotations
import random
from cards import preflop_key, evaluate
from equity import PREFLOP, estimate
from pushfold import probability as pushfold_probability


def _percentiles():
    ordered = sorted(PREFLOP, key=lambda k: PREFLOP[k][0], reverse=True)
    total = 0
    out = {}
    for k in ordered:
        n = 6 if len(k) == 2 else 4 if k.endswith('s') else 12
        out[k] = (total + n / 2) / 1326
        total += n
    return out


PERCENTILE = _percentiles()


class Agent:
    def __init__(self, seed=10, config=None):
        self.rng = random.Random(seed)
        self.config = dict(open_scale=1.15, aggression=1., call_margin=.025,
                           bluff_scale=1., range_penalty=1., pushfold=True)
        if config:
            self.config.update(config)
        self.reset()

    def reset(self):
        """Call on every new table; anonymized seat statistics stay table-local."""
        self.stats = {}
        self.hand = None
        self.seen = 0
        self.voluntary = set()
        self.pre_raisers = set()
        self.observed_preflop = set()
        self.equity_key = None
        self.equity_value = None
        self.failures = 0

    def _observe(self, o):
        hand = o.get('hand_id', 0)
        history = o.get('history', ())
        if hand != self.hand or len(history) < self.seen:
            self.hand, self.seen = hand, 0
            self.voluntary = set()
            self.pre_raisers = set()
            self.observed_preflop = set()
        # Stats: opportunities, voluntary actions, preflop raises, postflop
        # raises, postflop passive actions, folds. Shrink strongly to priors.
        for a in history[self.seen:]:
            s = self.stats.setdefault(a['seat'], [0, 0, 0, 0, 0, 0])
            kind = a['action']
            if a['street'] == 0:
                # Hero can fold before seeing later opponents act. Count only
                # observed preflop opportunities, never censored unknowns.
                if kind in ('fold','check','call','raise') and a['seat'] not in self.observed_preflop:
                    s[0] += 1
                    self.observed_preflop.add(a['seat'])
                if kind in ('call','raise') and a['seat'] not in self.voluntary:
                    s[1] += 1
                    self.voluntary.add(a['seat'])
                if kind=='raise' and a['seat'] not in self.pre_raisers:
                    s[2] += 1
                    self.pre_raisers.add(a['seat'])
            elif kind in ('raise', 'call', 'check'):
                s[3 if kind == 'raise' else 4] += 1
            s[5] += kind == 'fold'
        self.seen = len(history)

    def _style(self, seat):
        s = self.stats.get(seat, [0] * 6)
        # Shrink early seat estimates toward broad, non-team-specific priors.
        vpip = min(.85, (s[1] + 6) / (s[0] + 20))
        pfr = min(.65, (s[2] + 3.6) / (s[0] + 20))
        aggression = (s[3] + 3) / (s[3] + s[4] + 10)
        return vpip, pfr, aggression

    @staticmethod
    def _raise(o, target):
        if not o['can_raise']:
            return dict(action='call' if o['to_call'] else 'check')
        target = min(o['max_raise_to'], max(o['min_raise_to'], int(target)))
        return dict(action='raise', amount=target)

    def act(self, observation):
        """Return one legal local action, with check/fold on malformed input."""
        if not isinstance(observation,dict):
            self.failures += 1
            return dict(action='fold')
        try:
            return self._act(observation)
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError):
            self.failures += 1
            return dict(action='fold' if observation.get('to_call', 1) else 'check')

    def _act(self, o):
        cards = o['hole'] + o['board']
        if len(o['hole'])!=2 or len(o['board']) not in (0,3,4,5) or len(set(cards))!=len(cards) or any(type(c) is not int or not 0<=c<52 for c in cards):
            raise ValueError('invalid cards')
        self._observe(o)
        ps, hero = o['players'], o['hero']
        me = ps[hero]
        live = [p for p in ps if not p['folded'] and p['seat'] != hero]
        if not live:
            return dict(action='check')
        bb, call, pot = max(1, o['big_blind']), o['to_call'], max(1, o['pot'])
        stack, bet = me['stack'], me['bet']
        effective = min(stack, max(p['stack'] + max(0, p['bet'] - bet) for p in live))
        if not o['board']:
            return self._preflop(o, live, effective)
        return self._postflop(o, live, effective)

    def _risk_threshold(self, o, live, odds):
        """Exact local ICM threshold only for an isolated heads-up all-in call."""
        payouts = o.get('payouts')
        hero = o['hero']; me = o['players'][hero]; call = o['to_call']
        if not payouts or len(live) != 1 or call < me['stack']:
            return odds
        from icm import utility
        villain = live[0]['seat']
        stacks = [p['stack'] for p in o['players']]
        alive = sum(p['stack'] + p.get('committed',0)>0 for p in o['players'])
        # Exact ICM topology is precomputed for up to eight seats.
        if alive>8:
            return odds
        eligible = self._eligible_pot(o)
        folded = list(stacks); folded[villain] += o['pot']
        won = list(stacks); won[hero] += eligible; won[villain] += o['pot']-eligible
        # Hero's call is already in their remaining stack; win pot adds net pot.
        lost = list(stacks); lost[hero] -= call; lost[villain] += o['pot'] + call
        def u(s):
            if s[hero]==0:
                return payouts[alive-1] if len(payouts)>=alive else 0.
            return utility(s, payouts[:sum(x>0 for x in s)], hero)
        uf, uw, ul = u(folded), u(won), u(lost)
        return max(odds, min(.95, (uf-ul) / max(1e-12, uw-ul)))

    @staticmethod
    def _eligible_pot(o):
        me=o['players'][o['hero']]
        cap=me.get('committed',me['bet'])+o['to_call']
        return max(1,sum(min(p.get('committed',p['bet']),cap) for p in o['players']))

    def _preflop(self, o, live, effective):
        hero, ps = o['hero'], o['players']; me = ps[hero]
        bb, call, pot = max(1, o['big_blind']), o['to_call'], self._eligible_pot(o)
        key = preflop_key(o['hole']); q = PREFLOP[key]; rank = PERCENTILE[key]
        history = [a for a in o['history'] if a['street'] == 0]
        raises = [a for a in history if a['action'] == 'raise']
        limpers = sum(a['action'] == 'call' for a in history)
        active = [p['seat'] for p in ps if p['stack'] + p.get('committed', 0) > 0]
        # Players still to act after hero and before the big blind.
        bbseat = next((a['seat'] for a in history if a['action'] == 'big_blind'), (o['button']+2)%len(ps))
        behind = 0; s = (hero + 1) % len(ps)
        while s != (bbseat + 1) % len(ps):
            if s in active and not ps[s]['folded'] and ps[s]['stack'] > 0:
                behind += 1
            s = (s + 1) % len(ps)
        depth = effective / bb
        if not raises and not o['can_raise']:
            # A blind can itself be an all-in; this is still a priced call.
            odds=call/(pot+call) if call else 0.
            eq=q[0] ** (1+max(0,len(live)-1)*.72)
            return dict(action='call' if call and eq>self._risk_threshold(o,live,odds) else 'fold' if call else 'check')
        # Position-sensitive first-in ranges; no limping except completing SB.
        if not raises:
            if self.config['pushfold'] and not limpers and behind>=1 and depth<=12 and o['can_raise']:
                probability=pushfold_probability(key,(effective+me['bet'])/bb,behind)
                if self.rng.random()<probability:
                    return self._raise(o,o['max_raise_to'])
                return dict(action='fold' if call else 'check')
            fraction = (.78 if len(active)==2 else .52 if behind <= 1 else .39 if behind==2 else .29 if behind==3 else .23 if behind==4 else .19 if behind==5 else .16)
            fraction *= self.config['open_scale']
            fraction /= 1 + .20 * limpers
            if depth <= 12:
                jam_fraction = fraction * (.78 if depth > 8 else 1.05 if depth > 4 else 1.35)
                if rank < jam_fraction and o['can_raise']:
                    return self._raise(o, o['max_raise_to'])
                return dict(action='call' if call and call <= .15*bb else 'fold' if call else 'check')
            if rank < fraction and o['can_raise']:
                target = (2.3 if depth<40 else 2.6)*bb + limpers*bb
                if len(active)==2:
                    target=2.3*bb
                return self._raise(o, target)
            # Cheap speculative entry, with implied odds and sufficient depth.
            suited = key.endswith('s'); paired = len(key)==2
            connected = abs(o['hole'][0]//4-o['hole'][1]//4) <= 2
            if limpers and call <= bb and depth >= 24 and (paired or suited and connected and rank < .65):
                return dict(action='call' if call else 'check')
            if len(active)==2 and hero==o['button'] and call<=bb*.5 and rank<.88:
                return dict(action='call')
            return dict(action='fold' if call else 'check')
        last = raises[-1]; vpip,pfr,agg = self._style(last['seat'])
        nraises = len(raises)
        # Use the solved caller strategy only in its clean heads-up setting,
        # with nearly equal initial stacks and no supplied ICM awards.
        if self.config['pushfold'] and len(active)==2 and len(live)==1 and nraises==1 and not limpers and not o.get('payouts'):
            opponent=ps[last['seat']]
            villain_total=opponent['stack']+opponent.get('committed',opponent['bet'])
            hero_total=me['stack']+me.get('committed',me['bet'])
            if opponent['stack']==0 and villain_total<=15*bb and abs(hero_total-villain_total)<=.25*max(hero_total,villain_total):
                probability=pushfold_probability(key,villain_total/bb,1,0)
                return dict(action='call' if self.rng.random()<probability else 'fold')
        width = max(.055, min(.5, pfr * (1.4 if len(active)<4 else 1.)))
        if nraises>=2: width = max(.035, width * .35)
        # Table data range profiles: uniform, top35, top15, top5.
        if width >= .35: eq = q[1]+(q[0]-q[1])*min(1,(width-.35)/.65)
        elif width >= .15: eq = q[2]+(q[1]-q[2])*(width-.15)/.20
        elif width >= .05: eq = q[3]+(q[2]-q[3])*(width-.05)/.10
        else: eq = q[3]-.025
        opponents = sum(p['bet'] > me['bet'] and p['seat'] != hero and not p['folded'] for p in ps)
        eq = max(.01, eq) ** (1 + max(0, opponents-1)*.72)
        odds = call/(pot+call) if call else 0
        risk = self._risk_threshold(o, live, odds)
        # Value three/four-bets; broaden against observed loose aggression.
        premium = rank < (.026 if nraises==1 else .013)
        if pfr>.30 and nraises==1: premium = rank < .055
        if premium and o['can_raise']:
            target = max(o['min_raise_to'], me['bet'] + call + max(pot*.85, call*2.2))
            if depth<28 or target > me['bet'] + .40*me['stack']:
                target=o['max_raise_to']
            return self._raise(o,target)
        allin = call >= .65*me['stack'] or all(p['stack']==0 for p in live)
        if allin:
            threshold = risk + self.config['call_margin']
        else:
            position = (hero-o['button'])%len(ps)
            # Button and cutoff realize more equity than blinds.
            realization = .91 if position==0 else .82
            if len(key)==2 or key.endswith('s'): realization += .05
            threshold = odds / realization + self.config['call_margin']
            if call > 4*bb and rank > .23 and not key.endswith('s'):
                threshold += .08
            if nraises>=2 and call>.15*me['stack']: threshold += .04
        if eq > threshold:
            return dict(action='call' if call else 'check')
        # Set-mining needs sufficient odds; no calling a jam to chase a set.
        if len(key)==2 and nraises==1 and call<.055*effective and call<.10*pot:
            return dict(action='call')
        return dict(action='fold' if call else 'check')

    def _postflop(self, o, live, effective):
        hero=o['hero']; me=o['players'][hero]
        call, pot, bb = o['to_call'], self._eligible_pot(o), max(1,o['big_blind'])
        board=o['board']
        if len(board)==5:
            br=evaluate(board)
            cat=br>>20
            top=(br>>16)&15
            # Guaranteed shared nuts: folding can never improve chip EV.
            # No range transformation or independence approximation applies.
            shared = (cat==8 and top==12 or cat==7 and (((br>>12)&15)==12 or top==12 and ((br>>12)&15)==11) or
                      cat==4 and top==12 and max(sum(c%4==s for c in board) for s in range(4))<3)
            if shared:
                return dict(action='call' if call else 'check')
            hr=evaluate(o['hole']+board)
            if hr>>20==8 and (hr>>16)&15==12:
                return self._raise(o,o['max_raise_to']) if o['can_raise'] else dict(action='call' if call else 'check')
        key=(tuple(o['hole']),tuple(board))
        if key!=self.equity_key:
            self.equity_key=key
            self.equity_value=estimate(o['hole'],board)
        qs=self.equity_value
        history=o['history']; street=o['street']
        pre=[a for a in history if a['street']==0 and a['action']=='raise']
        street_raises=[a for a in history if a['street']==street and a['action']=='raise']
        mine_raised=any(a['seat']==hero for a in pre)
        equities=[]; aggressiveness=[]
        for p in live:
            vpip,pfr,agg=self._style(p['seat']); aggressiveness.append(agg)
            pr=sum(a['seat']==p['seat'] for a in pre)
            width=max(.12,min(.95,pfr if pr else vpip*1.5))
            if len(pre)>1: width=max(.08,width*.55)
            if width>=.6: eq=qs[1]+(qs[0]-qs[1])*(width-.6)/.4
            elif width>=.3: eq=qs[2]+(qs[1]-qs[2])*(width-.3)/.3
            else: eq=qs[3]+(qs[2]-qs[3])*max(0,(width-.12)/.18)
            # A bet or call narrows ranges toward stronger hands. Use a smooth
            # odds transformation, preserving the tails for draws and nuts.
            evidence=0.
            for a in history:
                if a['seat']!=p['seat'] or a['street']==0: continue
                if a['action']=='raise': evidence += .32 if a['street']==street else .22
                elif a['action']=='call': evidence += .14
            if street_raises and street_raises[-1]['seat']==p['seat']:
                ratio=call/max(1,pot-call)
                evidence += min(.8,ratio*.38) + (.2 if street==3 else 0)
            evidence *= self.config['range_penalty'] * max(.35,1.35-agg*1.7)
            eq=eq/max(.00001,eq+(1-eq)*(1+evidence*2.2))
            equities.append(eq)
        # Correlated showdown outcomes; geometric aggregation tempers the
        # independent-opponent product rather than treating it as exact.
        q=1.
        for eq in equities: q *= max(.001,eq)
        if len(equities)>1: q **= .82+.18/len(equities)
        odds=call/(pot+call) if call else 0
        threshold=self._risk_threshold(o,live,odds)
        n=len(live); spr=effective/max(pot,1)
        board=o['board']; ranks={c//4 for c in board}; counts=[sum(c%4==s for c in board) for s in range(4)]
        wet=max(counts)>=3 or len(ranks)>=3 and max(ranks)-min(ranks)<=5
        own_suits=[sum(c%4==s for c in o['hole']+board) for s in range(4)]
        flushdraw=street<3 and any(own_suits[s]==4 and any(c%4==s for c in o['hole']) for s in range(4))
        rnd=self.rng.random()
        value_threshold = (.68 if street==1 else .72 if street==2 else .76) + .025*max(0,n-1)
        if call:
            margin=self.config['call_margin'] + (.025 if street<3 and spr>2 else 0)
            if q < threshold+margin:
                return dict(action='fold')
            # Raise for clear value; avoid inflating middling hand pots.
            required = .80 if street<3 else .87
            if q>required and o['can_raise'] and rnd<min(1,.88*self.config['aggression']):
                target=me['bet']+call+int((pot+call)*(.80 if wet else .65))
                if spr<1.5 or target-me['bet']>.6*me['stack']:
                    target=o['max_raise_to']
                return self._raise(o,target)
            return dict(action='call')
        if not o['can_raise']:
            return dict(action='check')
        # Value sizing and low-frequency semibluffs. Multiway bluffs decline
        # quickly. Checkback medium equity to preserve showdown value.
        value=q>value_threshold
        in_position=not any(p['stack']>0 and (p['seat']-o['button'])%len(o['players']) > (hero-o['button'])%len(o['players']) for p in live) if hero!=o['button'] else True
        bluff=0.
        if n==1:
            if street==1 and mine_raised: bluff=.22 if wet else .34
            elif street==2 and flushdraw: bluff=.27
            elif street==3 and q<.25: bluff=.09
            elif in_position and q<.48: bluff=.13
        elif n==2 and street==1 and mine_raised and flushdraw: bluff=.12
        if value or rnd<bluff*self.config['bluff_scale']:
            fraction = (.60 if wet else .42) if street<3 else .65
            if q>.94: fraction=.85
            if not value: fraction=.42 if street==1 else .60
            target=me['bet']+max(bb,int(pot*fraction))
            if value and spr<1.35: target=o['max_raise_to']
            return self._raise(o,target)
        return dict(action='check')


PokerBot = Agent
