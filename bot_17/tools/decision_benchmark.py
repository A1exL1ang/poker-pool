"""Whole-Agent warm wall/CPU latency, errors, legal-action and fallback audit."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
start_import=time.perf_counter_ns()
from agent import Agent
import equity
from baselines import CallingAgent, PressureAgent, TightAgent, RandomAgent
from simulator import Table
from tournament import latency_summary
IMPORT_NS=time.perf_counter_ns()-start_import


def corpus():
    data=[]
    for seed in range(171701,171709):
        agents=[Agent(seed),CallingAgent(seed+1),PressureAgent(seed+2),TightAgent(seed+3),
                RandomAgent(seed+4),Agent(seed+5),PressureAgent(seed+6),TightAgent(seed+7)]
        table=Table(agents,seed=seed,max_hands=40,collect_observations=True,
                    payouts=[.5,.3,.2,0,0,0,0,0])
        table.run()
        data += [('natural',o) for o in table.observations]
    # Wide and tight multiway observations plus full-table ICM large calls.
    rng=random.Random(171717)
    for i in range(500):
        n = [2,4,8][i%3]
        cards=rng.sample(range(52),7)
        street=1+i%3
        board=cards[2:2+street+2]
        hero=dict(seat=0,stack=900,bet=10,committed=100,folded=False,all_in=False)
        players=[hero]+[dict(seat=s,stack=900,bet=200,committed=290,folded=False,all_in=False) for s in range(1,n)]
        if i%5==0:
            players=([hero]+[dict(seat=1,stack=0,bet=910,committed=1000,folded=False,all_in=True)]+
                     [dict(seat=s,stack=1000,bet=0,committed=0,folded=True,all_in=False) for s in range(2,8)])
            label='icm_stress'
        else:
            label='multiway_stress'
        call=max(p['bet'] for p in players)-hero['bet']
        o=dict(hand_id=10000+i,hero=0,button=1,street=street,hole=cards[:2],board=board,players=players,
               pot=sum(p['committed'] for p in players),big_blind=20,small_blind=10,to_call=call,
               min_raise_to=2*max(p['bet'] for p in players)-10,max_raise_to=910,
               can_raise=label!='icm_stress',legal_actions=['fold','call']+(['raise'] if label!='icm_stress' else []),
               payouts=[.5,.3,.2,0,0,0,0,0],history=[dict(seat=1,street=street,action='raise',amount=200,paid=200)])
        data.append((label,o))
    for i in range(100):
        cards=rng.sample(range(52),7)
        street=1+i%3
        hero=dict(seat=0,stack=1000,bet=0,committed=10,folded=False,all_in=False)
        players=[hero]+[dict(seat=s,stack=0,bet=b,committed=b+10,folded=False,all_in=True)
                        for s,b in enumerate([10,20,50,100,200,400,800],1)]
        data.append(('sidepot_stress',dict(hand_id=20000+i,hero=0,button=7,street=street,
             hole=cards[:2],board=cards[2:2+street+2],players=players,
             pot=sum(p['committed'] for p in players),big_blind=20,small_blind=10,to_call=800,
             min_raise_to=1200,max_raise_to=1000,can_raise=False,legal_actions=['fold','call'],
             payouts=[.5,.3,.2,0,0,0,0,0],
             history=[dict(seat=7,street=street,action='raise',amount=800,paid=800)])))
    rng.shuffle(data)
    return data


def legal(o,a):
    if not isinstance(a,dict) or a.get('action') not in o['legal_actions']:
        return False
    if a['action']=='raise':
        x=a.get('amount')
        return type(x) is int and max(p['bet'] for p in o['players']) < x <= o['max_raise_to'] and (x>=o['min_raise_to'] or x==o['max_raise_to'])
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calls',type=int,default=30000)
    p.add_argument('--output',type=Path,default=ROOT/'measurements'/'decision_latency.json')
    args=p.parse_args()
    observations=corpus()
    init=time.perf_counter_ns(); bot=Agent(17); init_ns=time.perf_counter_ns()-init
    for _,o in observations[:100]:
        bot.act(o)
    bot.reset()
    gc.collect()
    # Keep normal garbage collection enabled: do not hide runtime pauses.
    wall=[]; cpu=[]; grouped={}; errors=invalid=0
    equity.sampler_stats(reset=True)
    for i in range(args.calls):
        label,o=observations[i%len(observations)]
        wall_start=time.perf_counter_ns(); cpu_start=time.thread_time_ns()
        try:
            a=bot.act(o)
        except Exception:
            errors+=1; a={}
        elapsed_cpu=time.thread_time_ns()-cpu_start
        elapsed_wall=time.perf_counter_ns()-wall_start
        wall.append(elapsed_wall);cpu.append(elapsed_cpu)
        grouped.setdefault(label,[]).append(elapsed_wall)
        if not legal(o,a): invalid+=1
    result=dict(backend=equity.BACKEND,calls=args.calls,corpus_size=len(observations),
                platform=platform.platform(),python=platform.python_version(),gc_enabled=gc.isenabled(),
                module_import_ms=IMPORT_NS/1e6,constructor_us=init_ns/1e3,
                wall=latency_summary(wall),thread_cpu=latency_summary(cpu),
                thread_cpu_warning=('Windows per-thread accounting is coarse/quantized; use wall results.' if sys.platform=='win32' else None),
                by_group={k:latency_summary(v) for k,v in grouped.items()},
                errors=errors,invalid_actions=invalid,internal_fallbacks=bot.fallbacks,
                sampler=equity.sampler_stats(),
                source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['agent.py','equity.py','risk.py']},
                note='Whole in-process act. Warm after import. Normal GC; wall scheduling pauses included. Synthetic stress states mixed with local legal game observations.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
