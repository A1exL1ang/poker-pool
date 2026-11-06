"""Reproducible, seat-rotated LOCAL tournament and paired-reference evaluation.

Example (run from workspace):
  python benchmarks/evaluate.py --tables 200 --seed-start 100000 --split heldout \
      --lineup mixed --payout top3 --schedule doubling --output measurements/heldout.json

Payouts and blind increases are explicitly synthetic scenarios. Shared deal
seeds reduce variance but action-dependent eliminations can change future
card assignment. This is neither a contest score estimate nor a solver test.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import Agent, DEFAULTS
from baselines import EquityBot, make_lineup
from simulator import run_tournament

SOURCE_FILES=("agent.py","equity.py","icm.py","assets/equity_predict.py",
              "assets/equity_model.json","assets/preflop.json","assets/preflop_narrow.json",
              "assets/rank_scores.zlib","assets/__init__.py","baselines.py","simulator.py")


def source_hashes():
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in SOURCE_FILES if (ROOT/name).exists()}


SOURCE_HASHES_AT_IMPORT=source_hashes()


PAYOUTS = {
    "top3": (.5,.3,.2),
    "winner": (1.,),
    "top4": (.4,.3,.2,.1),
    "flat3": (1/3,1/3,1/3),
    "linear": (8/36,7/36,6/36,5/36,4/36,3/36,2/36,1/36),
}
SCHEDULES = {
    "doubling": ((10,20),(20,40),(40,80),(80,160),(160,320),(320,640),(640,1280),(1280,2560)),
    "gentle": ((10,20),(15,30),(25,50),(40,80),(60,120),(100,200),(150,300),(250,500)),
    "fast": ((10,20),(25,50),(60,120),(150,300),(350,700),(800,1600),(1600,3200),(3200,6400)),
}
SCENARIOS = {
    "mixed": ("mixed","top3","doubling"),
    "tag": ("tag","top3","doubling"),
    "nit": ("nit","top3","doubling"),
    "lag": ("lag","top3","doubling"),
    "maniac": ("maniac","top3","doubling"),
    "station": ("station","top3","doubling"),
    "shove": ("shove","top3","doubling"),
    "random": ("random","top3","doubling"),
    "selfplay": ("selfplay","top3","doubling"),
    "varied": ("varied","top3","doubling"),
    "slow": ("mixed","top3","gentle"),
    "winner": ("mixed","winner","doubling"),
    "flat": ("mixed","flat3","doubling"),
}


def summary(values):
    n = len(values)
    mean = statistics.fmean(values) if n else 0.
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.
    return dict(n=n,mean=mean,se=se,ci95=[mean-1.96*se,mean+1.96*se])


def latency_summary(values):
    ordered = sorted(values)
    def q(p):
        return ordered[min(len(ordered)-1, int((len(ordered)-1)*p))] / 1e3 if ordered else 0.
    return dict(decisions=len(ordered), p50_us=q(.5), p95_us=q(.95),
                p99_us=q(.99), p999_us=q(.999), max_us=q(1),
                overruns_1ms=sum(v > 1_000_000 for v in ordered),
                overrun_fraction=sum(v > 1_000_000 for v in ordered)/max(1,len(ordered)))


def balanced_hero_seat(index):
    # seed_start+i makes button cycle i%8. Rotate the offset too: each
    # physical seat and initial dealer-relative position appears once per64.
    return (index + index//8) % 8


def _case(args):
    index, seed, lineup, payouts, schedule, max_hands, params, reference_params, reference_style, limit = args
    seat = balanced_hero_seat(index)
    pair = []
    for which in ("candidate", "reference"):
        rivals = make_lineup(lineup, seed=seed*17)
        random.Random(seed ^ 0x12B0).shuffle(rivals)
        if which == "candidate":
            hero = Agent(seed=seed ^ 120012,params=params)
        elif reference_params is not None:
            hero = Agent(seed=seed ^ 120012,params=reference_params)
        else:
            hero = EquityBot(reference_style,seed=seed ^ 120012)
        rivals.insert(seat,hero)
        result = run_tournament(rivals,seed, max_hands=max_hands,
                                blind_schedule=schedule,payouts=payouts,
                                decision_limit_ns=1_000_000 if limit else None)
        pair.append(dict(points=result["points"][seat],
                         field_points=statistics.fmean(v for i,v in enumerate(result["points"]) if i!=seat),
                         chips=result["stacks"][seat],placement=result["placements"][seat],
                         outright_win=not result["capped"] and result["stacks"][seat]>0,
                         capped=result["capped"],hands=result["hands"],
                         invalid=result["invalid_actions"][seat],exceptions=result["exceptions"][seat],
                         timeouts=result["timeouts"][seat],latencies_ns=result["latencies_ns"][seat],
                         internal_fallbacks=getattr(hero,"fallbacks",0),
                         table_invalid=sum(result["invalid_actions"]),
                         table_exceptions=sum(result["exceptions"])))
    return dict(seed=seed,hero_seat=seat,initial_button=seed%8,candidate=pair[0],reference=pair[1],
                source_hashes_at_worker_import=SOURCE_HASHES_AT_IMPORT)


def evaluate(*,tables=200,seed_start=100000,lineup="mixed",payout="top3",
             schedule="doubling",max_hands=200,params=None,reference_params=None,
             reference_style="tag",workers=1,enforce_deadline=False,split="heldout"):
    if tables < 1 or not 1 <= workers <= 8 or not 0 <= max_hands <= 200:
        raise ValueError("require tables>=1, workers in1..8, max_hands in0..200")
    jobs = [(i,seed_start+i,lineup,PAYOUTS[payout],SCHEDULES[schedule],max_hands,
             params,reference_params,reference_style,enforce_deadline) for i in range(tables)]
    started = perf_counter()
    if workers == 1:
        results = [_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_case,jobs,chunksize=max(1,min(10,tables//(workers*4)))))
    aggregates = {}
    for which in ("candidate","reference"):
        rows = [r[which] for r in results]
        aggregates[which] = {
            "points":summary([r["points"] for r in rows]),
            "field_points":summary([r["field_points"] for r in rows]),
            "placement":summary([r["placement"] for r in rows]),
            "chips":summary([r["chips"] for r in rows]),
            "outright_win_rate":statistics.fmean(r["outright_win"] for r in rows),
            "cap_rate":statistics.fmean(r["capped"] for r in rows),
            "mean_hands":statistics.fmean(r["hands"] for r in rows),
            "invalid_actions":sum(r["invalid"] for r in rows),
            "exceptions":sum(r["exceptions"] for r in rows),
            "timeouts":sum(r["timeouts"] for r in rows),
            "internal_fallbacks":sum(r["internal_fallbacks"] for r in rows),
            "latency":latency_summary([v for r in rows for v in r["latencies_ns"]]),
            "all_table_invalid_actions":sum(r["table_invalid"] for r in rows),
            "all_table_exceptions":sum(r["table_exceptions"] for r in rows),
        }
    compact = []
    source_versions={}
    for row in results:
        hashes=row["source_hashes_at_worker_import"]
        version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        source_versions[version]=hashes
        compact.append({"seed":row["seed"],"hero_seat":row["hero_seat"],"initial_button":row["initial_button"],
                        "candidate_points":row["candidate"]["points"],
                        "reference_points":row["reference"]["points"],
                        "candidate_chips":row["candidate"]["chips"],
                        "reference_chips":row["reference"]["chips"],
                        "source_version":version})
    report_hashes=source_hashes()
    return dict(split=split,environment=dict(python=sys.version,platform=platform.platform(),
                processor=platform.processor()),seconds=perf_counter()-started,
                source_hashes_at_evaluator_import=SOURCE_HASHES_AT_IMPORT,
                source_hashes_at_report=report_hashes,source_versions_at_worker_import=source_versions,
                source_changed_during_process=any(h!=report_hashes for h in source_versions.values()),
                source_hash_note="Import-time hashes identify each worker's loaded checkpoint; report hashes detect later on-disk edits. Freeze source/assets before final evaluation.",
                config=dict(tables=tables,seed_start=seed_start,seed_stop_exclusive=seed_start+tables,
                hero_seat_rotation=True,initial_seat_dealer_offsets_balanced_per64=True,
                lineup=lineup,synthetic_payout=payout,payouts=PAYOUTS[payout],
                synthetic_schedule=schedule,blind_schedule=SCHEDULES[schedule],max_hands=max_hands,
                starting_stack=2000,hands_per_level=25,workers=workers,enforce_1ms_fallback=enforce_deadline,
                candidate_params=dict(DEFAULTS,**(params or {})),
                reference=(dict(kind="Agent",params=dict(DEFAULTS,**reference_params)) if reference_params is not None
                           else dict(kind="EquityBot",style=reference_style))),
                metrics=aggregates,
                paired_points_delta=summary([r["candidate"]["points"]-r["reference"]["points"] for r in results]),
                tables=compact,
                limitations=["Synthetic local payout and blind scenarios, not official contest values.",
                             "Transparent scripted opponents do not represent the unknown contest field.",
                             "Shared deck seeds; action-dependent elimination changes later card assignment.",
                             "Wall-clock timings include scheduler interference; hardware differs from contest.",
                             "Confidence intervals are asymptotic 1.96*sample-SE; no multiple-testing correction."])


def _read_params(path):
    if path is None:
        return None
    value=json.loads(Path(path).read_text())
    return value.get("params",value)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables",type=int,default=200)
    parser.add_argument("--seed-start",type=int,default=100000)
    parser.add_argument("--split",choices=("training","validation","heldout"),default="heldout")
    parser.add_argument("--phase",choices=("tuning","validation","heldout"),help="preferred evaluation phase alias")
    parser.add_argument("--scenario",action="append",help="repeat named scenario, 'all', or lineup:payout:schedule")
    parser.add_argument("--lineup",choices=("mixed","tag","nit","lag","maniac","station","shove","random","selfplay","varied"),default="mixed")
    parser.add_argument("--payout",choices=tuple(PAYOUTS),default="top3")
    parser.add_argument("--schedule",choices=tuple(SCHEDULES),default="doubling")
    parser.add_argument("--max-hands",type=int,default=200)
    parser.add_argument("--params-file","--params",dest="params_file")
    parser.add_argument("--reference-params-file")
    parser.add_argument("--reference-style",choices=("nit","tag","lag","maniac"),default="tag")
    parser.add_argument("--reference",help="baseline style, or JSON candidate-parameter path for paired comparison")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--enforce-deadline",action="store_true")
    parser.add_argument("--output",default="measurements/evaluation.json")
    args=parser.parse_args()
    phase=args.phase or ("tuning" if args.split=="training" else args.split)
    split="training" if phase=="tuning" else phase
    reference_params=_read_params(args.reference_params_file)
    reference_style=args.reference_style
    if args.reference:
        if args.reference in ("nit","tag","lag","maniac"):
            reference_style=args.reference
        else:
            reference_params=_read_params(args.reference)
    scenarios=[]
    for name in args.scenario or ():
        if name=="all":
            scenarios.extend((key,*value) for key,value in SCENARIOS.items())
        elif name in SCENARIOS:
            scenarios.append((name,*SCENARIOS[name]))
        else:
            fields=name.split(":")
            if len(fields)!=3 or fields[0] not in ("mixed","tag","nit","lag","maniac","station","shove","random","selfplay","varied") or fields[1] not in PAYOUTS or fields[2] not in SCHEDULES:
                parser.error("scenario must be a known name or lineup:payout:schedule")
            scenarios.append((name,*fields))
    if not scenarios:
        scenarios=[(f"{args.lineup}:{args.payout}:{args.schedule}",args.lineup,args.payout,args.schedule)]
    results=[]
    for index,(name,lineup,payout,schedule) in enumerate(scenarios):
        measured=evaluate(tables=args.tables,seed_start=args.seed_start+index*1_000_000,split=split,lineup=lineup,
                          payout=payout,schedule=schedule,max_hands=args.max_hands,
                          params=_read_params(args.params_file),reference_params=reference_params,
                          reference_style=reference_style,workers=args.workers,enforce_deadline=args.enforce_deadline)
        measured["phase"],measured["scenario"]=phase,name
        results.append(measured)
    result=results[0] if len(results)==1 else {"phase":phase,"results":results}
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("benchmark outputs must remain in this bot workspace")
    output.parent.mkdir(parents=True,exist_ok=True)
    raw_output=output.with_suffix(".trials.jsonl")
    with raw_output.open("w") as raw:
        for measured in results:
            for trial in measured["tables"]:
                raw.write(json.dumps(dict(phase=phase,scenario=measured["scenario"],**trial))+"\n")
    result["raw_trials_path"]=str(raw_output)
    output.write_text(json.dumps(result,indent=2))
    for measured in results:
        print(json.dumps({"output":str(output),"scenario":measured["scenario"],"phase":phase,
                          "seconds":measured["seconds"],"config":measured["config"],
                          "metrics":measured["metrics"],"paired_points_delta":measured["paired_points_delta"]},indent=2))


if __name__ == "__main__":
    main()
