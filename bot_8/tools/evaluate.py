"""Seat-balanced local tournaments against original reference heuristics."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from agent import Agent
from tools.baselines import Baseline
from tools.simulator import Table, LAB_POINTS, LAB_BLINDS

FIELDS={"mixed":["abc","lag","station","nit","abc","lag","station","abc"],
        "abc":["abc"]*8,"aggressive":["lag"]*8,"passive":["station"]*8,"tight":["nit"]*8}


def stats(values):
    avg=statistics.fmean(values)
    se=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else 0
    return {"mean":avg,"se":se,"ci95":[avg-1.96*se,avg+1.96*se]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tables",type=int,default=200)
    p.add_argument("--seed",type=int,default=80000)
    p.add_argument("--field",choices=FIELDS,default="mixed")
    p.add_argument("--schedule",choices=["lab","winner","flat"],default="lab")
    p.add_argument("--scale",type=float,default=.5)
    p.add_argument("--budget-us",type=int,default=450,help="0 disables adaptive wall-clock guard")
    p.add_argument("--enforce-deadline",action="store_true",help="locally fold/check actions exceeding 1 ms wall time")
    p.add_argument("--control",action="store_true")
    p.add_argument("--output",default="evidence/tournaments.json")
    args=p.parse_args()
    points=LAB_POINTS if args.schedule=="lab" else (1.,0.,0.,0.,0.,0.,0.,0.) if args.schedule=="winner" else (1.,1.,1.,0.,0.,0.,0.,0.)
    results=[]; controls=[]; wins=0; actions=hands=0; capped=0; timed_calls=timed_out=0
    start=time.perf_counter()
    for i in range(args.tables):
        hero=i%8
        agents=[Baseline(style) for style in FIELDS[args.field]]
        bot=Agent(sample_scale=args.scale,time_budget_us=args.budget_us or None)
        if args.enforce_deadline:
            class Timed:
                def reset(self,table_id): bot.reset(table_id)
                def act(self,o):
                    nonlocal timed_calls,timed_out
                    from kestrel.model import Action
                    begin=time.perf_counter_ns(); a=bot.act(o); elapsed=time.perf_counter_ns()-begin
                    timed_calls+=1
                    if elapsed>1000000:
                        timed_out+=1
                        return Action("fold" if o.to_call else "check")
                    return a
            agents[hero]=Timed()
        else:
            agents[hero]=bot
        result=Table(agents,seed=args.seed+i,points=points).run()
        results.append(result["points"][hero]); wins+=result["finishes"][hero]==1
        actions+=result["actions"]; hands+=result["hands"]; capped+=result["hands"]>=200
        if args.control:
            agents=[Baseline(style) for style in FIELDS[args.field]]
            control=Table(agents,seed=args.seed+i,points=points).run()
            controls.append(control["points"][hero])
        if (i+1)%100==0:
            print(f"{i+1}/{args.tables}: mean points {statistics.fmean(results):.4f}",flush=True)
    out={"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"configuration":vars(args),
         "local_only":True,"points":points,"blind_levels":LAB_BLINDS,"hands_per_level":25,"start_stack":2000,
         "score":stats(results),"equal_share_score":sum(points)/8,"sole_winner_fraction":wins/args.tables,
         "table_hands":hands,"all_player_decisions":actions,"tables_reaching_cap":capped,"seconds":time.perf_counter()-start,
         "timed_hero_calls":timed_calls,"hero_timeout_fallbacks":timed_out,
         "limitations":"Original simple reference opponents; invented experimental point/blind schedules beyond known initial blinds. This is not official compatibility or a prediction of contest rank."}
    if controls:
        out["control_score"]=stats(controls)
        out["paired_point_difference"]=stats([a-b for a,b in zip(results,controls)])
    Path(args.output).write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":
    main()
