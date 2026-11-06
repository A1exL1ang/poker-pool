"""Frozen submissions, imported under private package names; no policy edits.

Adapter conversion is included in the reported decision latency for every bot.
Only public observations and the acting player's cards cross this boundary.
"""
import importlib
import importlib.util
import json
import sys
from itertools import count
from pathlib import Path
from common.engine import Action

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {1:'pokerbot',2:'pokerbot',3:'bot3',4:'bot',5:'pokerbot',6:'bot6',7:'bot7',8:'kestrel'}
STREETS = ('preflop','flop','turn','river')
_TABLE_IDS = count(1)

def load_package(alias, directory):
    if alias not in sys.modules:
        spec = importlib.util.spec_from_file_location(alias, directory/'__init__.py', submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
    return alias

def history(o):
    # Reconstruct facing costs and pot-before from PUBLIC events and blind posts.
    bets = [0]*o.n_seats
    remaining = list(o.starting_stacks)
    pot = 0
    for i in range(o.n_seats):
        paid = min(remaining[i],o.ante) if o.active[i] else 0
        remaining[i] -= paid; pot += paid
    for i, blind in ((o.sb_seat,o.sb),(o.bb_seat,o.bb)):
        paid = min(remaining[i],blind); remaining[i] -= paid; bets[i] += paid; pot += paid
    street = 0; current = o.bb
    folded = [not active for active in o.active]
    result=[]
    for j,e in enumerate(o.history):
        if e.street != street:
            street=e.street; bets=[0]*o.n_seats; current=0
        opponents=[i for i in range(o.n_seats) if i!=e.seat and not folded[i]]
        target=current if any(remaining[i]>0 for i in opponents) else max((bets[i] for i in opponents),default=0)
        facing=max(0,target-bets[e.seat])
        result.append(dict(id=o.hand_number*100000+j,seat=e.seat,street=STREETS[e.street],action=e.kind,
                           amount=e.amount,to_call=facing,pot_before=pot,paid=e.paid,hand_id=o.hand_number,facing_bet=facing>0))
        if e.kind=='fold': folded[e.seat]=True
        if e.kind in ('raise','call'):
            bets[e.seat]=e.amount; remaining[e.seat]-=e.paid; pot+=e.paid
            current=max(current,e.amount)
    return result

class Adapter:
    def __init__(self, number):
        self.number=number
        alias=load_package('frozen_bot_'+str(number),ROOT/f'bot_{number}'/PACKAGES[number])
        self.module=importlib.import_module(alias+('.policy' if number in (4,8) else '.agent'))
        self.alias=alias
        self.calls=0
        self.internal_errors=0

    def reset(self, seed, seat, n_seats):
        self.bot=self.module.Agent() if self.number==8 else self.module.Agent(seed=seed)
        self.table_id='evaluation-table-'+str(next(_TABLE_IDS))
        self.calls=0; self.internal_errors=0

    def act(self,o):
        n=self.number
        call=min(o.to_call,o.stacks[o.seat])
        legal=['fold','call'] if call else ['check']
        if o.can_raise: legal.append('raise')
        h=history(o)
        players=[dict(seat=i,stack=o.stacks[i],bet=o.street_bets[i],committed=o.contributions[i],folded=o.folded[i],all_in=o.all_in[i]) for i in range(o.n_seats)]
        # Remove already busted seats only where the documented contract expects
        # remaining players. Keep folded contributors throughout each hand.
        if n in (1,4,6,7,8): players=[p for p in players if o.active[p['seat']]]
        payouts=o.payouts[:len(players)]
        d=dict(table_id=self.table_id,hand_id=o.hand_number,hero=o.seat,button=o.button,
               hole_cards=o.hole,board=o.board,players=players,pot=o.pot,big_blind=o.bb,
               to_call=call,min_raise_to=o.min_raise_to,max_raise_to=o.max_raise_to,
               can_raise=o.can_raise,legal_actions=legal,payouts=payouts,history=h)
        if n==1:
            d['hole_cards']=tuple((c%4)*13+c//4 for c in o.hole)
            d['board']=tuple((c%4)*13+c//4 for c in o.board)
            d['legal_actions']=[{'action':k,**({'min':o.min_raise_to,'max':o.max_raise_to} if k=='raise' else {'amount':call} if k=='call' else {})} for k in legal]
        elif n==2:
            d.pop('hole_cards');d.pop('legal_actions')
            d.update(hole=o.hole,legal=tuple(legal),hands_remaining=o.hands_left)
            d['players']=[dict(seat=i,stack=o.stacks[i],street_bet=o.street_bets[i],invested=o.contributions[i],folded=o.folded[i],sitting_out=not o.active[i]) for i in range(o.n_seats)]
            d['history']=[dict(seat=e.seat,street=e.street,kind=e.kind,amount=e.amount) for e in o.history]
        elif n==3:
            d.update(hero_seat=o.seat,street=STREETS[o.street],current_bet=o.street_bets[o.seat]+o.to_call)
        elif n==4:
            d.pop('hole_cards');d.pop('legal_actions');d.update(hole=o.hole,ante=o.ante,to_call=o.to_call)
            d['players']=[dict(seat=p['seat'],stack=p['stack'],committed=p['bet'],contributed=p['committed'],folded=p['folded'],all_in=p['all_in']) for p in players]
            d['history']=[{k:e[k] for k in ('seat','street','action','amount','to_call','pot_before')} for e in h]
        elif n==5:
            d['legal_actions']={k:({'min':o.min_raise_to,'max':o.max_raise_to} if k=='raise' else call if k=='call' else True) for k in legal}
            d['history']=[dict(e,player=e['seat']) for e in h]
        elif n==6:
            d['players']=[dict(seat=p['seat'],stack=p['stack'],bet=p['bet'],contributed=p['committed'],folded=p['folded'],all_in=p['all_in']) for p in players]
        elif n==7:
            d.pop('hole_cards');d.pop('can_raise');d.pop('history');d['hole']=o.hole
            d['events']=[dict(id=e['id'],seat=e['seat'],street=e['street'],action=e['action'],amount=e['paid'] if e['action']=='call' else e['amount'],to_call=e['to_call']) for e in h]
        elif n==8:
            d.pop('hole_cards');d.pop('legal_actions');d.update(hole=o.hole,to_call=o.to_call,hands_remaining=o.hands_left)
            d['players']=[dict(seat=p['seat'],stack=p['stack'],committed=p['bet'],invested=p['committed'],folded=p['folded'],all_in=p['all_in']) for p in players]
            d['history']=[dict(id=e['id'],seat=e['seat'],street=e['street'],action=e['action'],amount=e['paid'] if e['action']=='call' else e['amount'],to_call=e['to_call'],pot_before=e['pot_before']) for e in h]
        self.calls+=1
        previous_info=getattr(self.bot,'last_info',None)
        a=self.bot.act_dict(d) if n==7 else self.bot.act(d)
        last_error=getattr(self.bot,'last_error',None)
        info=getattr(self.bot,'last_info',None)
        # Bot1 creates a fresh diagnostic dictionary for each caught input error.
        # A later early return may retain it: count new reports, not stale state.
        reported_fallback=(n==1 and isinstance(info,dict) and info is not previous_info
                           and bool(info.get('fallback')))
        if last_error or reported_fallback:
            self.internal_errors+=1
        if last_error:
            raise RuntimeError(last_error)
        if not isinstance(a,dict): a=a.as_dict()
        return Action(a['action'],int(a.get('amount',0)))

class Baseline:
    def __init__(self,variant='ppo'):
        if variant not in ('ppo','net'):raise ValueError(variant)
        self.path=(ROOT/'evaluation/baseline_net' if variant=='net' else ROOT/'evaluation/baseline_repo/vibe_coding_exp/claude_exp_1/submission')
        self.alias=load_package('frozen_claude_'+variant,self.path/'nlbot')
        self.engine=importlib.import_module(self.alias+'.engine')
        self.build=importlib.import_module(self.alias+'.bot.registry').build
    def reset(self,seed,seat,n_seats):
        c=json.loads((self.path/'submission_config.json').read_text())
        kind=c.pop('kind');kw=c.pop('heuristic_params',{})
        self.bot=self.build(kind,**c,**kw)
        self.bot.reset(seed,seat,n_seats)
    def act(self,o):
        d={k:getattr(o,k) for k in o.__dataclass_fields__}
        d['history']=tuple(self.engine.Event(e.seat,e.street,e.kind,e.amount,e.paid) for e in o.history)
        a=self.bot.act(self.engine.Observation(**d))
        return Action(a.kind,a.amount)
