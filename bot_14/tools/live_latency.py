"""Per-seat timing attribution during real simulated play of the locked bot.

The benchmark wrapper adds timer/collection overhead outside Agent.decide.
Engine timeout counts include that overhead and are reported separately.
This diagnostic never fits or changes the policy.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter_ns, process_time_ns
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pokerbot.game import Table
from pokerbot.opponents import Opponent
from pokerbot.strategy import Agent
from tools.benchmark import BLINDS, FIELDS, PAYOUTS
from tools.confirm import REFERENCE
from tools.latency import stats
from tools.package import ROOT, hashes


class TimedAgent:
    def __init__(self,seed,config):
        self.bot=Agent(seed=seed,config=config)
        self.wall=defaultdict(list);self.cpu=defaultdict(list)

    def reset(self):
        self.bot.reset();self.wall.clear();self.cpu.clear()

    def decide(self,observation):
        c0=process_time_ns();w0=perf_counter_ns()
        action=self.bot.decide(observation)
        w=perf_counter_ns()-w0;c=process_time_ns()-c0
        self.wall[observation["street"]].append(w)
        self.cpu[observation["street"]].append(c)
        return action


def play(task):
    seed,field,config=task
    rng=random.Random(seed);seat=rng.randrange(8)
    hero=TimedAgent(seed*101+seat,config)
    styles=list(FIELDS[field]);rng.shuffle(styles)
    agents=[]
    for i in range(8):
        if i==seat:agents.append(hero)
        else:
            style=styles.pop()
            agents.append(Agent(seed*101+i,REFERENCE) if style=="candidate" else Opponent(style,seed*101+i))
    table=Table(seed=seed,strict=True,decision_timeout_ms=1.,payouts=PAYOUTS["top3"])
    own_timeouts=other_timeouts=0
    for hand in range(200):
        table.set_blinds(*BLINDS["moderate"][hand//25])
        result=table.play_hand(agents)
        for error in result.errors:
            if error["kind"]=="timeout":
                if error["seat"]==seat:own_timeouts+=1
                else:other_timeouts+=1
        if sum(s>0 for s in table.stacks)<=1:break
    return dict(wall=dict(hero.wall),cpu=dict(hero.cpu),hero_engine_timeouts=own_timeouts,
        other_engine_timeouts=other_timeouts,illegal=table.invalid_actions,exceptions=table.exceptions,
        internal_budget_fallbacks=hero.bot.diagnostics["budget_fallbacks"],
        native_fallbacks=hero.bot.diagnostics["equity_native_fallbacks"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables",type=int,default=1200)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--seed",type=int,default=3014014)
    p.add_argument("--config",default="{}")
    p.add_argument("--lock",default="measurements/locked_candidate.json")
    p.add_argument("--output",default="measurements/live_latency_linux.json")
    args=p.parse_args()
    if not 1<=args.workers<=8 or not 1<=args.tables<=4000:p.error("bounded tables/workers required")
    lock=json.loads((ROOT/args.lock).read_text())
    if hashes()!=lock["files"]:raise ValueError("runtime differs from candidate lock")
    config=dict(lock["config"],**json.loads(args.config))
    fields=("mixed","rollout","aggressive","tight","loose","self")
    tasks=[(args.seed+i,fields[i%len(fields)],config) for i in range(args.tables)]
    wall=defaultdict(list);cpu=defaultdict(list)
    totals={k:0 for k in ("hero_engine_timeouts","other_engine_timeouts","illegal","exceptions","internal_budget_fallbacks","native_fallbacks")}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(play,tasks,chunksize=10):
            for k in totals:totals[k]+=result[k]
            for street,values in result["wall"].items():wall[street].extend(values)
            for street,values in result["cpu"].items():cpu[street].extend(values)
    if hashes()!=lock["files"]:raise ValueError("runtime changed during diagnostic")
    report=dict(phase="post-lock live timing diagnostic, no fitting",tables=args.tables,workers=args.workers,
        seed_start=args.seed,config=config,platform=platform.platform(),python=sys.version,
        wall=stats(sum(wall.values(),[])),cpu=stats(sum(cpu.values(),[])),
        wall_by_street={k:stats(v) for k,v in wall.items()},cpu_by_street={k:stats(v) for k,v in cpu.items()},
        counts=totals,notes="Hero timers measure Agent.decide only; per-seat engine timeouts additionally include wrapper timer/collection overhead. All OS scheduling delays count. Eight concurrent workers stress the allocated CPUs. Synthetic fields/payouts/blinds.")
    path=ROOT/args.output;path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
