"""Fresh development confirmation across opponents and scoring sensitivities."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark import play, summary

REFERENCE = dict(open_scale=1., call_margin=.025, value_threshold=.64,
    bluff_scale=1., shove_depth=11., icm=True, adaptation=True, model=True,
    nash=True, native=True, native_samples=256, native_mix=1., budget_ns=650000)


def paired(rows, base):
    d = [x["points"]-y["points"] for x,y in zip(rows,base)]
    mean = statistics.mean(d)
    se = statistics.stdev(d)/len(d)**.5 if len(d)>1 else 0.
    return dict(mean_difference=mean,se=se,ci95_normal=(mean-1.96*se,mean+1.96*se))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", required=True, help="JSON mapping names to partial config overrides")
    p.add_argument("--tables", type=int, default=1000)
    p.add_argument("--sensitivity-tables", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=144000)
    p.add_argument("--output", default="measurements/confirmation.json")
    args = p.parse_args()
    if not 1<=args.workers<=8 or not 1<=args.tables<=4000 or not 1<=args.sensitivity_tables<=4000:
        p.error("bounded workers/tables required")
    overrides=json.loads(Path(args.configs).read_text())
    configs={"reference":REFERENCE}
    configs.update({k:dict(REFERENCE,**v) for k,v in overrides.items() if k!="reference"})
    cases=[(field,"top3","moderate",args.tables) for field in ("mixed","rollout","aggressive","self")]
    cases += [("mixed","winner","fast",args.sensitivity_tables), ("mixed","flat3","slow",args.sensitivity_tables), ("mixed","linear","moderate",args.sensitivity_tables), ("rollout","winner","fast",args.sensitivity_tables), ("rollout","flat3","slow",args.sensitivity_tables), ("tight","top3","fast",args.sensitivity_tables)]
    report=dict(phase="fresh development confirmation, not held-out",seed=args.seed,configs=configs,cases=cases,results={})
    start=time.perf_counter()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name,config in configs.items():
            report["results"][name]={}
            differences=[]
            for index,(field,payouts,blinds,tables) in enumerate(cases):
                key=f"{field}/{payouts}/{blinds}"
                tasks=[(args.seed+index*10000+i,field,payouts,blinds,config,REFERENCE) for i in range(tables)]
                rows=list(pool.map(play,tasks,chunksize=12))
                cell=dict(summary=summary(rows),rows=rows)
                if name!="reference":
                    base=report["results"]["reference"][key]["rows"]
                    cell["paired"]=paired(rows,base)
                    differences.extend([x["points"]-y["points"] for x,y in zip(rows,base)])
                report["results"][name][key]=cell
                report["seconds"]=time.perf_counter()-start
                output.write_text(json.dumps(report,indent=2))
                print(json.dumps(dict(config=name,case=key,summary=cell["summary"],paired=cell.get("paired"))),flush=True)
            if differences:
                se=statistics.stdev(differences)/len(differences)**.5
                report["results"][name]["aggregate_paired"]=dict(mean_difference=statistics.mean(differences),se=se,ci95_normal=(statistics.mean(differences)-1.96*se,statistics.mean(differences)+1.96*se))
    report["seconds"]=time.perf_counter()-start
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v.get("aggregate_paired") for k,v in report["results"].items()}),flush=True)


if __name__=="__main__":main()
