"""Reproducible synthetic eight-seat tournaments; no official scoring claims."""
import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from agent import Agent
from engine import play_hand
from icm import utilities
from opponents import ProfileBot
from tools.reference_policy import Agent as ReferenceAgent
from tools.challenge_opponents import ChallengeBot

SCHEDULES = {'normal':[20,40,80,120,200,300,500,800],
             'slow':[20,30,40,60,80,120,160,240],
             'fast':[20,50,100,200,400,800,1600,3200]}
PAYOUTS = {'winner':[1,0,0,0,0,0,0,0],
           'top3':[.5,.3,.2,0,0,0,0,0],
           'linear':[7/28,6/28,5/28,4/28,3/28,2/28,1/28,0]}
LINEUPS = {'mixed':['tight','loose','station','maniac','balanced','nit','random'],
           'tight':['tight','balanced','nit','tight','balanced','nit','tight'],
           'loose':['loose','station','maniac','loose','station','maniac','random'],
           'self':['self']*7,
           'challenge':['solid','pressure','sticky','rock','solid','pressure','sticky']}


def tournament(seed, config=None, lineup='mixed', schedule='normal', capture=False, enforce_time=False, reference_hero=False, policy_class=None):
    rng=random.Random(seed)
    seat=rng.randrange(8)
    profiles=LINEUPS[lineup][:]; rng.shuffle(profiles)
    agents=[]
    for i in range(8):
        if i==seat: agents.append((policy_class or (ReferenceAgent if reference_hero else Agent))(seed*11+i,config))
        else:
            name=profiles.pop()
            agents.append(ReferenceAgent(seed*11+i) if name=='self' else ChallengeBot(name,seed*11+i) if lineup=='challenge' else ProfileBot(name,seed*11+i))
    stacks=[2000]*8; button=rng.randrange(8)
    awards={name:[0.]*8 for name in PAYOUTS}
    active=8; errors=[]; timings=[]; obs=[]; dealt=0
    if capture:
        real=agents[seat]
        class Recorder:
            def act(self,o):
                if len(obs)<1000: obs.append(o)
                return real.act(o)
            @property
            def failures(self): return real.failures
        agents[seat]=Recorder()
    for hand in range(200):
        if active<2: break
        before=stacks[:]
        bb=SCHEDULES[schedule][hand//25]
        result=play_hand(agents,stacks,button,bb//2,bb,seed*1009+hand,hand_id=hand,enforce_time=enforce_time)
        dealt+=1; stacks=result['stacks']; timings.extend(result['decisions_ns']); errors.extend(result['errors'])
        eliminated=sorted(result['eliminations'],key=lambda i:before[i])
        while eliminated:
            group=[i for i in eliminated if before[i]==before[eliminated[0]]]
            for name,payout in PAYOUTS.items():
                award=sum(payout[active-len(group):active])/len(group)
                for i in group: awards[name][i]=award
            active-=len(group)
            eliminated=[i for i in eliminated if i not in group]
        button=next((button+j)%8 for j in range(1,9) if stacks[(button+j)%8]>0)
    for name,payout in PAYOUTS.items():
        remaining=utilities(stacks,payout[:active])
        for i,x in enumerate(remaining): awards[name][i]+=x
        assert abs(sum(awards[name])-1)<1e-9
    failures=sum(getattr(a,'failures',0) for a in agents)
    return {'seed':seed,'hero':seat,'scores':{k:v[seat] for k,v in awards.items()},
            'stacks':stacks,'hands':dealt,'errors':errors,'internal_fallbacks':failures,
            'decision_count':len(timings),'over_1ms':sum(t>1_000_000 for t in timings),
            'max_us':max(timings,default=0)/1000,'observations':obs}


def summarize(rows):
    scores={}
    for k in PAYOUTS:
        values=[r['scores'][k] for r in rows]
        mean=statistics.mean(values)
        se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0
        scores[k]={'mean':mean,'se':se,'ci95':[mean-1.96*se,mean+1.96*se]}
    return {'tables':len(rows),'scores':scores,'hands':sum(r['hands'] for r in rows),
            'decision_count':sum(r['decision_count'] for r in rows),'over_1ms':sum(r['over_1ms'] for r in rows),
            'max_us':max(r['max_us'] for r in rows),'errors':sum(len(r['errors']) for r in rows),
            'internal_fallbacks':sum(r['internal_fallbacks'] for r in rows)}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--tables',type=int,default=100); p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--lineup',choices=LINEUPS,default='mixed'); p.add_argument('--schedule',choices=SCHEDULES,default='normal')
    p.add_argument('--config',default='{}'); p.add_argument('--output',default='results/benchmark.json');p.add_argument('--capture',action='store_true');p.add_argument('--enforce-time',action='store_true')
    a=p.parse_args(); start=time.time()
    rows=[tournament(a.seed+i,json.loads(a.config),a.lineup,a.schedule,a.capture,a.enforce_time) for i in range(a.tables)]
    report={'settings':vars(a),'seconds':time.time()-start,'summary':summarize(rows),'rows':rows}
    Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report['summary'],indent=2))


if __name__=='__main__': main()
