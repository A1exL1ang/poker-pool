"""Unseen-policy evaluation family, constructed separately from Agent logic.

Shares card/equity estimates but has different action rules. Still synthetic;
this does not represent an external competitive poker population.
"""
import random
from equity import estimate,PREFLOP
from cards import preflop_key


class ChallengeBot:
    def __init__(self,style='solid',seed=0):
        self.style=style;self.rng=random.Random(seed)

    def act(self,o):
        me=o['players'][o['hero']];call=o['to_call'];pot=max(1,o['pot']);bb=o['big_blind']
        n=sum(not p['folded'] for p in o['players'])-1
        loose={'rock':-.07,'solid':0.,'pressure':.06,'sticky':.04}[self.style]
        def raise_to(target):
            return dict(action='raise',amount=min(o['max_raise_to'],max(o['min_raise_to'],int(target))))
        passive=dict(action='call' if call else 'check')
        if not o['board']:
            q=PREFLOP[preflop_key(o['hole'])]
            raised=sum(a['action']=='raise' for a in o['history'])
            if not raised:
                limit=(.58 if n>=5 else .55 if n>=3 else .50)-loose
                if q[0]>limit and o['can_raise']:
                    return raise_to(o['max_raise_to'] if me['stack']<9*bb else (2.1 if self.style=='pressure' else 3.)*bb)
                return passive if not call or call<.4*bb and q[0]>.40 else dict(action='fold')
            odds=call/(pot+call)
            eq=(q[1] if self.style in ('pressure','sticky') else q[2])**(1+max(0,n-2)*.35)
            if q[3]>.57-loose*.5 and o['can_raise']:
                return raise_to(o['max_raise_to'] if me['stack']<30*bb else me['bet']+call+pot)
            required=odds+(.06 if call>.4*me['stack'] else .14)-loose
            return passive if eq>required else dict(action='fold' if call else 'check')
        q=estimate(o['hole'],o['board'])[1 if self.style=='sticky' else 2]
        q=q**(1+max(0,n-1)*.75)
        odds=call/(pot+call) if call else 0
        if call:
            q-=.04 if self.style=='pressure' else .10
            if q<odds+.015-loose:
                return dict(action='fold')
            if q>.83-loose and o['can_raise'] and self.rng.random()<.75:
                return raise_to(me['bet']+call+pot)
            return passive
        value=q>(.62 if o['street']<3 else .70)-loose
        bluff=self.rng.random()<(.27 if self.style=='pressure' else .07 if self.style=='rock' else .15)/max(1,n)
        if o['can_raise'] and (value or bluff):
            fraction=.33 if self.style=='pressure' else .85 if self.style=='sticky' else .66
            target=me['bet']+max(bb,pot*fraction)
            if value and me['stack']<pot:target=o['max_raise_to']
            return raise_to(target)
        return passive
