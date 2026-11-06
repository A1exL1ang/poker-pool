"""Real decision trace + synthetic stress wall/CPU timing, no benchmark IO in act."""
import argparse
import gc
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent import Agent
from baselines import Baseline,POOLS,make_baseline
from equity_model import asset_status
from sim import Tournament
from benchmarks.run import PAYOUTS,source_hashes


def summary(ns):
    ns=sorted(ns)
    return {'count':len(ns),'mean_us':statistics.mean(ns)/1000,
            **{f'p{p}_us':ns[min(len(ns)-1,int(len(ns)*p/100))]/1000 for p in (50,90,99,99.9)},
            'max_us':ns[-1]/1000,'over_1ms':sum(t>1000000 for t in ns)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tables',type=int,default=160)
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--params')
    p.add_argument('--output',default='measurements/latency.json')
    a=p.parse_args()
    params=json.loads(Path(a.params).read_text()) if a.params else None
    observations=[]
    actual=[]
    fallback=0
    def collect(obs,action,elapsed):
        if obs['hero']==0:
            observations.append(obs)
            actual.append(elapsed)
    for i in range(a.tables):
        pool=list(POOLS)[i%len(POOLS)]
        agents=[Agent(seed=1919+i,params=params)]+[Agent(seed=1931+i+j) if name=='mirror' else make_baseline(name,i*13+j) for j,name in enumerate(POOLS[pool])]
        Tournament(agents,seed=19800000+i,payouts=PAYOUTS[['chip','top3','linear'][i%3]],observer=collect).run()
        fallback+=agents[0].fallbacks
    # Include every observed history; each independent act below has cold
    # hand-history state (but imports/assets loaded), a harder parsing case.
    rng=random.Random(193818)
    rng.shuffle(observations)
    wall,cpu=[],[]
    by_street={k:[] for k in ('preflop','flop','turn','river')}
    overruns=[]
    for _ in range(a.repeat):
        bot=Agent(seed=19,params=params)
        for index,obs in enumerate(observations):
            # Table/hand reset is outside the decision clock. Each shuffled
            # observation now really starts with empty history and profiles.
            bot.reset()
            c0=time.thread_time_ns()
            t0=time.perf_counter_ns()
            action=bot.act(obs)
            elapsed=time.perf_counter_ns()-t0
            used=time.thread_time_ns()-c0
            wall.append(elapsed);cpu.append(used);by_street[obs['street']].append(elapsed)
            if elapsed>1000000:
                overruns.append({'index':index,'street':obs['street'],'history_length':len(obs['history']),
                                 'wall_us':elapsed/1000,'thread_cpu_us':used/1000})
            fallback+=bot.fallbacks
    report={'arguments':vars(a),'hashes':source_hashes(),'asset_status':asset_status(),
            'python':platform.python_version(),'platform':platform.platform(),
            'clock_boundary':'in-process act; excludes imports, model loading, observation construction and JSON IO',
            'gc_enabled':gc.isenabled(),'trace_decisions':summary(actual),'replay_wall':summary(wall),
            'thread_cpu_clock_note':'Windows per-call thread CPU values may be quantized; use wall measurements for the deadline',
            'replay_thread_cpu':summary(cpu),'by_street':{k:summary(v) for k,v in by_street.items() if v},
            'internal_fallbacks':fallback,'overrun_details':overruns}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='overrun_details'},indent=2))


if __name__=='__main__':
    main()
