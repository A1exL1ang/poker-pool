"""Bounded paired-seed development experiments; never used as held-out data."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark import play, summary

BASE = dict(native_mix=1., native_samples=256)
CONFIGS = dict(
    base=BASE,
    hybrid=dict(BASE, native_mix=.5),
    rollout=dict(BASE, native_mix=0., native_samples=384),
    portable=dict(BASE, native=False),
    value55=dict(BASE, value_threshold=.55),
    value60=dict(BASE, value_threshold=.60),
    value69=dict(BASE, value_threshold=.69),
    value74=dict(BASE, value_threshold=.74),
    call00=dict(BASE, call_margin=.0),
    call05=dict(BASE, call_margin=.05),
    call08=dict(BASE, call_margin=.08),
    open85=dict(BASE, open_scale=.85),
    open115=dict(BASE, open_scale=1.15),
    open130=dict(BASE, open_scale=1.30),
    bluff0=dict(BASE, bluff_scale=.0),
    bluff05=dict(BASE, bluff_scale=.5),
    bluff17=dict(BASE, bluff_scale=1.7),
    shove7=dict(BASE, shove_depth=7.),
    shove15=dict(BASE, shove_depth=15.),
    no_icm=dict(BASE, icm=False),
    no_adapt=dict(BASE, adaptation=False),
    no_nash=dict(BASE, nash=False),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", type=int, default=400)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=142000)
    p.add_argument("--configs", default="all")
    p.add_argument("--fields", default="mixed,rollout,aggressive,self")
    p.add_argument("--output", default="measurements/tuning.json")
    args = p.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.tables <= 4000:
        p.error("bounded workers1..8 and tables1..4000 required")
    selected = list(CONFIGS) if args.configs == "all" else args.configs.split(",")
    if "base" not in selected:
        selected.insert(0, "base")
    fields = args.fields.split(",")
    start = time.perf_counter()
    report = dict(phase="development/tuning", tables_per_cell=args.tables, seed_start=args.seed, configs={k:CONFIGS[k] for k in selected}, fields=fields, reference_config=BASE, results={}, selection={})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in selected:
            report["results"][name] = {}
            for field in fields:
                tasks = [(args.seed+i, field, "top3", "moderate", CONFIGS[name], BASE) for i in range(args.tables)]
                rows = list(pool.map(play, tasks, chunksize=12))
                cell = dict(summary=summary(rows), rows=rows)
                if name != "base":
                    ref = report["results"]["base"][field]["rows"]
                    differences = [r["points"] - b["points"] for r,b in zip(rows,ref)]
                    se = statistics.stdev(differences) / args.tables ** .5 if args.tables > 1 else 0.
                    cell["paired"] = dict(mean_difference=statistics.mean(differences), se=se, ci95_normal=(statistics.mean(differences)-1.96*se, statistics.mean(differences)+1.96*se))
                report["results"][name][field] = cell
                print(json.dumps(dict(config=name, field=field, summary=cell["summary"], paired=cell.get("paired"))), flush=True)
                report["seconds"] = time.perf_counter()-start
                output.write_text(json.dumps(report, indent=2))
            if name != "base":
                differences = [d for field in fields for d in [r["points"]-b["points"] for r,b in zip(report["results"][name][field]["rows"], report["results"]["base"][field]["rows"])]]
                se = statistics.stdev(differences) / len(differences)**.5
                report["selection"][name] = dict(mean_paired_difference=statistics.mean(differences), se=se, conservative_score=statistics.mean(differences)-se)
    report["seconds"] = time.perf_counter()-start
    report["strategy_sha256"] = hashlib.sha256((Path(__file__).resolve().parents[1]/"pokerbot"/"strategy.py").read_bytes()).hexdigest()
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(selection=report["selection"], seconds=report["seconds"])), flush=True)


if __name__ == "__main__":
    main()
