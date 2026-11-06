"""Untouched evaluation of a source/asset-locked candidate. Never fits a policy."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.benchmark import play, summary
from tools.confirm import REFERENCE
from tools.confirm import paired
from tools.package import ROOT, hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lock",default="measurements/locked_candidate.json")
    p.add_argument("--tables",type=int,default=2000)
    p.add_argument("--sensitivity-tables",type=int,default=500)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--seed",type=int,default=1014000)
    p.add_argument("--timeout-ms",type=float,default=1.)
    p.add_argument("--output",default="measurements/heldout_tournaments.json")
    args=p.parse_args()
    if not 1<=args.workers<=8 or not 1<=args.tables<=10000 or not 1<=args.sensitivity_tables<=4000:
        p.error("bounded workers/tables required")
    lock_path=ROOT/args.lock
    lock=json.loads(lock_path.read_text())
    if hashes()!=lock["files"]:raise ValueError("runtime differs from locked candidate")
    cases=[(f,"top3","moderate",args.tables) for f in ("mixed","rollout","aggressive","tight","loose","self")]
    for payout in ("top3","winner","linear","flat3"):
        for blind in ("moderate","fast","slow"):
            if (payout,blind)!=("top3","moderate"):
                cases.append(("mixed",payout,blind,args.sensitivity_tables))
    report=dict(phase="untouched held-out evaluation after source/asset lock",lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),config=lock["config"],reference=REFERENCE,seed_start=args.seed,timeout_ms=args.timeout_ms,synthetic_assumptions=True,results={})
    start=time.perf_counter();output=ROOT/args.output
    output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index,(field,payout,blind,tables) in enumerate(cases):
            key=f"{field}/{payout}/{blind}"
            tasks=[(args.seed+10000*index+i,field,payout,blind,lock["config"],REFERENCE,args.timeout_ms) for i in range(tables)]
            rows=list(pool.map(play,tasks,chunksize=12))
            cell=dict(summary=summary(rows),rows=rows,seed_start=args.seed+10000*index)
            if index in (0,1,2,5):
                reference_tasks=[(args.seed+10000*index+i,field,payout,blind,REFERENCE,REFERENCE,args.timeout_ms) for i in range(tables)]
                baseline=list(pool.map(play,reference_tasks,chunksize=12))
                cell["reference_summary"]=summary(baseline)
                cell["reference_rows"]=baseline
                cell["paired_vs_reference"]=paired(rows,baseline)
            report["results"][key]=cell;report["seconds"]=time.perf_counter()-start
            output.write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(case=key,summary=cell["summary"])),flush=True)
    if hashes()!=lock["files"]:raise ValueError("runtime changed during evaluation")
    report["seconds"]=time.perf_counter()-start
    output.write_text(json.dumps(report,indent=2))


if __name__=="__main__":main()
