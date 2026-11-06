"""Decision-only latency benchmark over real, varied local tournament states."""
import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from agent import Agent
from tools.simulator import Table, action_is_legal
from tools.baselines import Baseline


def summary(times):
    x=sorted(times)
    def pct(p): return x[min(len(x)-1,int((len(x)-1)*p))]/1000
    return {"calls":len(x),"mean_us":statistics.fmean(x)/1000,"p50_us":pct(.5),"p95_us":pct(.95),
            "p99_us":pct(.99),"p999_us":pct(.999),"max_us":max(x)/1000,
            "over_1ms":sum(v>1000000 for v in x)}


def collect(tables,scale,budget):
    corpus=[]
    for seed in range(tables):
        seat=seed%8
        agents=[Baseline(s) for s in ["abc","lag","station","nit","abc","lag","station","abc"]]
        agents[seat]=Agent(sample_scale=scale,time_budget_us=budget)
        def capture(o,a):
            if o.hero==seat:
                corpus.append(o)
        Table(agents,seed=70000+seed,capture=capture).run()
    return corpus


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tables",type=int,default=30)
    p.add_argument("--repeat",type=int,default=2)
    p.add_argument("--scale",type=float,default=.5)
    p.add_argument("--budget-us",type=int,default=450,help="0 disables adaptive wall-clock guard")
    p.add_argument("--output",default="evidence/latency.json")
    args=p.parse_args()
    budget=args.budget_us or None
    corpus=collect(args.tables,args.scale,budget)
    typed=[]; typed_cpu=[]; by_street={s:[] for s in ["preflop","flop","turn","river"]}
    for rep in range(args.repeat):
        bot=Agent(sample_scale=args.scale,time_budget_us=budget)
        for o in corpus:
            start=time.perf_counter_ns(); cpu=time.process_time_ns()
            a=bot.act(o)
            elapsed_cpu=time.process_time_ns()-cpu; elapsed=time.perf_counter_ns()-start
            assert action_is_legal(o,a)
            typed.append(elapsed); typed_cpu.append(elapsed_cpu); by_street[o.street].append(elapsed)
    dictionaries=[asdict(o) for o in corpus]
    dict_times=[]
    bot=Agent(sample_scale=args.scale,time_budget_us=budget)
    for d in dictionaries:
        start=time.perf_counter_ns(); bot.act(d); dict_times.append(time.perf_counter_ns()-start)
    result={"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"python":platform.python_version(),
            "platform":platform.platform(),"configuration":vars(args),"unique_observations":len(corpus),
            "typed_wall":summary(typed),"typed_process_cpu":summary(typed_cpu),"dict_wall":summary(dict_times),
            "by_street_wall":{k:summary(v) for k,v in by_street.items() if v},
            "note":"Includes state observation and action safety gate. Excludes import, process startup, and JSON serialization. Wall-clock outliers include OS scheduling; CPU clock resolution may be coarse."}
    Path(args.output).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
