"""Warm in-process replay timing, including validation and policy state updates."""
import argparse
import json
import platform
from pathlib import Path
import random
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from agent import Agent
from engine import Hand


def percentile(v,p): return v[min(len(v)-1,int((len(v)-1)*p))]/1000


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',default='results/development_capture.json');p.add_argument('--rounds',type=int,default=10);p.add_argument('--output',default='results/latency.json');p.add_argument('--model');a=p.parse_args()
    if a.model:
        import equity
        trees=json.loads(Path(a.model).read_text())['trees']
        equity.TREES=[(t['left'],t['right'],t['feature'],t['threshold'],t['value']) for t in trees]
    data=json.loads(Path(a.input).read_text())
    observations=[o for row in data['rows'] for o in row['observations']]
    agent=Agent(10010); buckets={'preflop':[],'postflop':[],'icm':[]};fallback=0;illegal=0
    start=time.perf_counter()
    for repeat in range(a.rounds):
        agent.reset()
        for o in observations:
            before=time.perf_counter_ns(); action=agent.act(o);duration=time.perf_counter_ns()-before
            buckets['postflop' if o['board'] else 'preflop'].append(duration)
            kind=action['action'];me=o['players'][o['hero']]
            legal = kind in ('call','fold') or kind=='check' and not o['to_call'] or kind=='raise' and o['can_raise'] and action['amount']<=o['max_raise_to'] and (action['amount']>=o['min_raise_to'] or action['amount']==o['max_raise_to'])
            illegal+=not legal
        fallback+=agent.failures
    # Explicit 8-survivor ICM all-in paths with user-supplied synthetic payouts.
    from copy import deepcopy
    template=next(o for o in observations if not o['board'])
    for i in range(2000):
        o=deepcopy(template);o['hero']=0;o['hole']=[48,45];o['board']=[];o['street']=0;o['hand_id']=i
        o['players']=[dict(seat=j,stack=0 if j==1 else 1000,bet=0 if j>1 else 1000 if j==1 else 0,committed=1000 if j==1 else 0,folded=j>1,all_in=j==1) for j in range(8)]
        o.update(pot=1000,to_call=1000,can_raise=False,min_raise_to=2000,max_raise_to=1000,payouts=[.5,.3,.2,0,0,0,0,0],history=[dict(seat=1,street=0,action='raise',amount=1000)])
        before=time.perf_counter_ns();agent.act(o);buckets['icm'].append(time.perf_counter_ns()-before)
    summary={}
    for name,values in buckets.items():
        if not values: continue
        values.sort();summary[name]={'calls':len(values),'p50_us':percentile(values,.5),'p95_us':percentile(values,.95),'p99_us':percentile(values,.99),'p999_us':percentile(values,.999),'max_us':values[-1]/1000,'over_1ms':sum(x>1_000_000 for x in values)}
    report={'platform':platform.platform(),'python':sys.version,'model':a.model or 'assets/equity_model.json','input_observations':len(observations),'rounds':a.rounds,'seconds':time.perf_counter()-start,'summary':summary,'internal_fallbacks':fallback,'illegal_actions':illegal,'note':'Wall-clock includes OS scheduling and GC; import/model initialization excluded. No hard real-time guarantee.'}
    Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
