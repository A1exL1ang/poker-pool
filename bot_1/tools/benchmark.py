"""End-to-end warm act latency with adversarial workload categories."""
import argparse
import gc
import datetime
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pokerbot.agent import Agent
from pokerbot.native import LIB,LOAD_ERROR


def positions(n=256,seed=6071):
    rng=random.Random(seed)
    out=[]
    for i in range(n):
        count=rng.choice((2,3,4,8))
        street=rng.choice((0,3,4,5))
        cs=rng.sample(range(52),2+street)
        cost=rng.choice((0,20,100,500,1500))
        ps=[{"seat":j,"stack":1500 if j==0 else 1200,"bet":0 if j==0 else cost,"committed":500 if j==0 else cost+500} for j in range(count)]
        pot=500*count+cost*(count-1)
        payouts=None
        if i%11==0:
            count=8
            ps=[{"seat":j,"stack":1500 if j!=1 else 0,"bet":0 if j!=1 else 1500,"committed":500 if j!=1 else 2000,"folded":j>=2} for j in range(count)]
            cost=1500
            pot=5500
            payouts=[50,30,20,0,0,0,0,0] if i%22 else [8,7,6,5,4,3,2,1]
        if i%17==0:
            # Closed eight-way all-in response with seven distinct payout
            # pools, stressing eligibility masks and both equity backends.
            count=8
            commitments=[500,2000,150,300,500,800,1000,1400]
            ps=[{"seat":j,"stack":9350 if j==0 else 0,"bet":1500 if j==1 else 0,"committed":commitments[j]} for j in range(count)]
            cost=1500
            pot=sum(commitments)
            payouts=None
        obs={"table_id":"latency","hand_id":i,"hero":0,"button":count-1,"players_to_act":count-1,
             "hole_cards":cs[:2],"board":cs[2:],"players":ps,"pot":pot,"big_blind":20,
             "to_call":cost,"legal_actions":[{"action":"fold"},{"action":"call","amount":cost}]+([] if cost==1500 else [{"action":"raise","min":max(20,cost*2),"max":1500}]) if cost else [{"action":"check"},{"action":"raise","min":20,"max":1500}]}
        if payouts:
            obs["payouts"]=payouts
        if i%17==0:
            obs["players_to_act"]=0
            obs["legal_actions"]=[{"action":"fold"},{"action":"call","amount":1500}]
        if i%7==0:
            obs["history"]=[{"id":i*100+j,"seat":j%count,"hand_id":i,"street":"preflop","action":"raise" if j%3==0 else "call"} for j in range(16)]
        out.append(obs)
    return out


def run(iterations=20000):
    corpus=positions()
    bot=Agent()
    for obs in corpus:
        bot.act(obs)
    results={"preflop":[],"postflop":[],"icm":[]}
    cpu_start=time.thread_time_ns()
    # Leave cyclic GC enabled: normal gameplay allocations are part of timing.
    for i in range(iterations):
        obs=corpus[i%len(corpus)]
        obs["hand_id"]=i
        obs["table_id"]=f"latency-{i//200}"
        for j,event in enumerate(obs.get("history",[])):
            event["hand_id"]=i
            event["id"]=i*32+j
        label="icm" if "payouts" in obs else "postflop" if obs["board"] else "preflop"
        start=time.perf_counter_ns()
        bot.act(obs)
        results[label].append((time.perf_counter_ns()-start)/1000)
    cpu_total_us=(time.thread_time_ns()-cpu_start)/1000
    def stats(xs):
        s=sorted(xs)
        return {"n":len(s),"mean_us":round(statistics.mean(s),3),"p50_us":round(s[len(s)//2],3),"p95_us":round(s[int(len(s)*.95)],3),"p99_us":round(s[int(len(s)*.99)],3),"p999_us":round(s[min(len(s)-1,int(len(s)*.999))],3),"max_us":round(max(s),3),"over_1ms":sum(x>1000 for x in s)}
    root=Path(__file__).resolve().parents[1]
    paths=[root/"agent.py",*sorted((root/"pokerbot").glob("*.py")),root/"pokerbot"/"fast_equity.cpp",root/"assets"/"preflop.json",Path(__file__).resolve()]
    if LIB is not None:
        paths.append(Path(LIB._name))
    return {"measured_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"python":sys.version,"platform":platform.platform(),"processor":platform.processor(),"native_accelerator":LIB is not None,"native_load_error":LOAD_ERROR,"gc_enabled":gc.isenabled(),"iterations":iterations,"scope":"warm Agent.act wall latency, including card parsing, legal adapter, model update, equity, optional ICM and periodic table resets; timer overhead included. Excludes process startup, imports and JSON transport. Wall-clock outliers are retained.","source_sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},"all":stats(sum(results.values(),[])),"categories":{k:stats(v) for k,v in results.items()},"thread_cpu_batch":{"total_us":cpu_total_us,"mean_us_per_decision_including_harness":cpu_total_us/iterations,"note":"Batch average only. Windows thread CPU clock advances in coarse accounting ticks; per-call CPU percentiles would be misleading."}}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--iterations",type=int,default=20000)
    p.add_argument("--output",default="benchmarks/latency.json")
    args=p.parse_args()
    result=run(args.iterations)
    Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
