"""Replay actual legal observations plus large-payout ICM stress states.

Reports wall and process CPU time independently; NEVER removes outliers. Process
CPU measurements separate scheduling delay diagnostically, not to excuse a
wall-clock timeout. Initialization and JSON decoding are outside act timing.
"""
import argparse
import io
import json
import platform
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter_ns, process_time_ns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent import Agent
from sim.benchmark import run, quantile, source_hashes
from sim.engine import legal_action


def stats(values):
    return {'n':len(values),'p50':quantile(values,.5)/1000,'p95':quantile(values,.95)/1000,
            'p99':quantile(values,.99)/1000,'p999':quantile(values,.999)/1000,
            'max':max(values,default=0)/1000,'over_1ms':sum(v>1_000_000 for v in values)}


def icm_states(count):
    rng=random.Random(181811)
    for k in range(count):
        cards=rng.sample(range(52),7)
        bet=rng.randrange(300,1700)
        hero_stack=rng.randrange(bet,4000)
        ps=[{'seat':i,'stack':rng.randrange(150,4000),'bet':0,'invested':0,
             'folded':i not in (0,1),'all_in':False} for i in range(8)]
        ps[0]['stack']=hero_stack
        ps[0]['invested']=100
        ps[1].update(stack=0,bet=bet,invested=bet+100,all_in=True)
        yield {'hand_id':10000+k,'hero':0,'button':7,'small_blind':10,'big_blind':20,
               'street':'river','hole':cards[:2],'board':cards[2:],'players':ps,
               'pot':bet+200,'to_call':bet,'min_raise_to':2*bet,'max_raise_to':hero_stack,
               'can_raise':False,'payouts':[35,25,18,12,6,3,1,0],
               'history':[{'seat':1,'street':'river','kind':'raise','amount':bet,'raise_to':bet}]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tables',type=int,default=100)
    p.add_argument('--stress',type=int,default=2000)
    p.add_argument('--config',default='{}')
    p.add_argument('--output',default='measurements/latency.json')
    args=p.parse_args()
    cfg=json.loads(args.config)
    buffer=io.StringIO()
    provenance=run(kwargs={'config':cfg},tables=args.tables,seed=187300,suite='mixed',payout='linear',
                   capture=buffer,capture_limit=100000)
    rows=[json.loads(line) for line in buffer.getvalue().splitlines()]
    agent=Agent(seed=18,config=cfg)
    wall,cpu=[],[]
    by_street={s:[] for s in ('preflop','flop','turn','river','icm_stress')}
    by_cpu={s:[] for s in by_street}
    last_table=None
    allstates=[(r['table_seed'],r['observation']['street'],r['observation']) for r in rows]
    allstates += [(0,'icm_stress',o) for o in icm_states(args.stress)]
    fallbacks=0
    for table,kind,obs in allstates:
        if table!=last_table:
            fallbacks+=agent.errors
            agent.reset()
            last_table=table
        c0=process_time_ns();w0=perf_counter_ns()
        action=agent.act(obs)
        w1=perf_counter_ns();c1=process_time_ns()
        legal_action(obs,action)
        wall.append(w1-w0);cpu.append(c1-c0)
        by_street[kind].append(w1-w0);by_cpu[kind].append(c1-c0)
    fallbacks+=agent.errors
    report={'platform':platform.platform(),'python':sys.version,'config':cfg,
            'model_and_source_sha256':source_hashes(),
            'wall_us':stats(wall),'process_cpu_us':stats(cpu),
            'wall_by_street_us':{k:stats(v) for k,v in by_street.items()},
            'cpu_by_street_us':{k:stats(v) for k,v in by_cpu.items()},
            'internal_fallbacks':fallbacks,'illegal_actions':0,
            'observation_corpus_tables':args.tables,'observations':len(rows),
            'limitations':['Initialization and JSON decoding excluded; actual act call measured.',
                           'CPU timing includes timer instrumentation; Windows CPU clock resolution can be coarse.',
                           'Wall-time overruns retained; no hard realtime guarantee.']}
    Path(args.output).parent.mkdir(exist_ok=True,parents=True)
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
