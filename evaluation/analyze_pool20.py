"""Audit complete balanced twenty-bot runs and bootstrap field-seed clusters.

Run substantive data analysis on a task-owned compute allocation, not the laptop.
Bootstrap resampling preserves all seats/payouts of each independent field seed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from schedule20 import PAYOUTS, VERSION, iter_matches, metadata

COUNTERS=("decisions","timeouts","errors","illegal","internal_errors",
          "safety_fallbacks","budget_fallbacks","native_fallbacks","total_latency_ns")
FAULTS=("errors","illegal","internal_errors")
PAYOUT_NAMES=tuple(PAYOUTS)


def validate_row(row, expected):
    """Fail closed on schedule drift, corrupt arrays, conservation, or faults."""
    if row.get("version") != VERSION: raise ValueError("wrong harness version")
    for key,value in expected.items():
        if row.get(key)!=value: raise ValueError(f"schedule mismatch {row.get('match_id')}: {key}")
    for key in (*COUNTERS,"points","final_stacks","placements","max_latency_ns"):
        values=row.get(key)
        if not isinstance(values,list) or len(values)!=8: raise ValueError(f"invalid {key} shape")
        if not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in values):
            raise ValueError(f"nonfinite or nonnumeric {key}")
        if min(values)<0: raise ValueError(f"negative {key}")
    for key in (*COUNTERS,"final_stacks","max_latency_ns"):
        if any(int(x)!=x for x in row[key]): raise ValueError(f"nonintegral {key}")
    if sum(row["final_stacks"])!=16000: raise ValueError("chip conservation")
    if abs(sum(row["points"])-1)>1e-9: raise ValueError("prize conservation")
    if any(x<1 or x>8 for x in row["placements"]): raise ValueError("invalid placement")
    for name in ("timeouts","errors","illegal"):
        if any(x>d for x,d in zip(row[name],row["decisions"])): raise ValueError(f"{name} exceeds decisions")
    if any(any(row[k]) for k in FAULTS):
        raise ValueError(f"unexplained fault in {row['match_id']}: " + str({k:row[k] for k in FAULTS}))


def bootstrap_ratios(sums, counts, replicates=5000, seed=2026110620, batch_size=64):
    """Ratio estimator under ordinary cluster bootstrap, shared across columns.

    Inputs C x D contain complete cluster totals and observation counts. Zero
    counts for absent entrants are required; dropping these clusters would lose
    covariance across comparisons. No individual decision/table resampling.
    """
    sums=np.asarray(sums,dtype=float);counts=np.asarray(counts,dtype=float)
    if sums.shape!=counts.shape or sums.ndim!=2: raise ValueError("cluster matrix shape")
    if len(sums)<2 or np.any(counts.sum(axis=0)<=0): raise ValueError("insufficient clusters/counts")
    if replicates<1 or batch_size<1: raise ValueError("invalid bootstrap size")
    if not np.isfinite(sums).all() or not np.isfinite(counts).all() or np.any(counts<0):
        raise ValueError("invalid cluster values")
    rng=np.random.default_rng(seed);n=len(sums)
    samples=np.empty((replicates,sums.shape[1]),dtype=float)
    for start in range(0,replicates,batch_size):
        size=min(batch_size,replicates-start)
        weights=rng.multinomial(n,np.full(n,1/n),size=size).astype(float)
        numer=weights@sums;denom=weights@counts
        if np.any(denom<=0): raise ValueError("empty bootstrap statistic; insufficient exposure")
        samples[start:start+size]=numer/denom
    return samples


def summary_stat(sums,counts,samples):
    total=float(np.sum(counts))
    return {"mean":float(np.sum(sums)/total),
            "ci95":[float(x) for x in np.quantile(samples,[.025,.975])],
            "tables":int(total),"field_seed_clusters":int(np.count_nonzero(counts))}


def analyze(paths,repetitions=4,replicates=5000):
    expected={r["match_id"]:r for r in iter_matches(repetitions)}
    cluster_ids={c:sorted({r["cluster_id"] for r in expected.values() if r["cohort"]==c})
                 for c in ("pool","mixed")}
    cluster_index={c:{x:i for i,x in enumerate(ids)} for c,ids in cluster_ids.items()}
    # Pool columns: bots1..20. Mixed: means1..22, bot1..20 minusPPO,
    # bot1..20 minusnet, then PPO-minusnet. Each has one column per payout.
    nstats={"pool":20,"mixed":63}
    sums={c:np.zeros((len(cluster_ids[c]),3,nstats[c])) for c in nstats}
    counts={c:np.zeros_like(sums[c]) for c in nstats}
    totals={str(n):{k:0 for k in COUNTERS} | {"max_latency_ns":0,"tables":0} for n in range(1,23)}
    by_cohort={c:{str(n):{k:0 for k in COUNTERS} | {"max_latency_ns":0,"tables":0}
                  for n in (range(1,21) if c=="pool" else range(1,23))} for c in nstats}
    seen=set();source_files={};missing_metadata=[];cohort_tables=Counter()
    for path in paths:
        path=Path(path);digest=hashlib.sha256()
        meta_path=path.with_suffix(path.suffix+".meta.json")
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            if meta.get("version")!=VERSION or meta.get("pilot") or meta.get("repetitions")!=repetitions:
                raise ValueError(f"incompatible shard metadata: {meta_path}")
            if meta.get("engine_assumptions",{}).get("timing_ns")!=1_000_000 or meta.get("engine_assumptions",{}).get("strict_timing") is not True:
                raise ValueError("shard does not declare 1ms timing")
        else:
            raise ValueError(f"missing shard metadata: {meta_path}")
        with path.open("rb") as f:
            for line_number,line in enumerate(f,1):
                digest.update(line)
                if not line.strip(): raise ValueError(f"blank row {path}:{line_number}")
                row=json.loads(line);mid=row.get("match_id")
                if mid in seen: raise ValueError(f"duplicate match {mid}")
                if mid not in expected: raise ValueError(f"unexpected match {mid}")
                validate_row(row,expected[mid]);seen.add(mid)
                c=row["cohort"];ci=cluster_index[c][row["cluster_id"]];pi=PAYOUT_NAMES.index(row["payout"])
                cohort_tables[c]+=1
                point=dict(zip(row["seats"],row["points"]))
                for seat,n in enumerate(row["seats"]):
                    sums[c][ci,pi,n-1]+=point[n];counts[c][ci,pi,n-1]+=1
                    for target in (totals[str(n)],by_cohort[c][str(n)]):
                        target["tables"]+=1
                        for key in COUNTERS:target[key]+=int(row[key][seat])
                        target["max_latency_ns"]=max(target["max_latency_ns"],int(row["max_latency_ns"][seat]))
                if c=="mixed":
                    for n in row["seats"]:
                        if n>20:continue
                        sums[c][ci,pi,22+n-1]+=point[n]-point[21];counts[c][ci,pi,22+n-1]+=1
                        sums[c][ci,pi,42+n-1]+=point[n]-point[22];counts[c][ci,pi,42+n-1]+=1
                    sums[c][ci,pi,62]+=point[21]-point[22];counts[c][ci,pi,62]+=1
        source_files[path.name]={"sha256":digest.hexdigest(),"bytes":path.stat().st_size}
    missing=set(expected)-seen
    if missing:raise ValueError(f"incomplete schedule: missing {len(missing)}, example={sorted(missing)[:3]}")
    for c in nstats:
        if not np.all(counts[c][:,:,0:nstats[c]] .sum(axis=0)>0):raise ValueError("empty statistic")
        # Every appearing entrant contributes all eight rotations per payout;
        # this also guards accidental partial-cluster bootstrap inputs.
        if not set(np.unique(counts[c])).issubset({0.,8.}):raise ValueError("partial seed cluster")
    samples={c:bootstrap_ratios(sums[c].reshape(len(sums[c]),-1),
                                counts[c].reshape(len(counts[c]),-1),replicates,
                                seed=2026110620+(c=="mixed")).reshape(replicates,3,nstats[c]) for c in nstats}
    result={"version":VERSION,"tables":len(seen),"cohort_tables":dict(cohort_tables),
            "audit":{"complete_unique_schedule":True,"correct_seats_and_seeds":True,
                     "chip_and_prize_conservation":True,"faults":0,"source_files":source_files,
                     "missing_shard_metadata":missing_metadata},
            "schedule":metadata(repetitions),"bootstrap_replicates":replicates,
            "method":"Percentile bootstrap of independent field-seed clusters within cohort, preserving all seat rotations and payouts. Per-statistic ratio of resampled point totals to appearance counts; absent entrants have zero counts. Shared resample weights preserve covariance across payouts and comparisons. Marginal 95% intervals; no multiplicity adjustment.",
            "limitations":["Synthetic common payouts/blinds and adapter interface; no official compatibility claim.",
                "Balanced field multiset gives equal pair exposure but does not uniformly sample all possible eight-player fields; higher-order opponent interactions may change rankings.",
                "Intervals describe variation under this field/deal sampling scheme and shared-hardware timing; they are not simultaneous rank guarantees or proof of universal strength.",
                "Same deck seeds need not produce identical actions because wall-clock fallback and time-budgeted policies are nondeterministic.",
                "Safety, budget and native fallback diagnostics are separate counters and may overlap; do not sum them into a distinct-failure total.",
                "Reported placing ranks at the hand cap are chip-order diagnostics, not expected ICM finishing places; rankings use points."],
            "payouts":{},"timing":totals,"timing_by_cohort":by_cohort}
    for pi,payout in enumerate(PAYOUT_NAMES):
        r={"pool_only":{},"mixed_fields":{},"paired_vs_ppo":{},"paired_vs_net":{}}
        for n in range(1,21):
            r["pool_only"][str(n)]=summary_stat(sums["pool"][:,pi,n-1],counts["pool"][:,pi,n-1],samples["pool"][:,pi,n-1])
            for label,column in (("paired_vs_ppo",22+n-1),("paired_vs_net",42+n-1)):
                r[label][str(n)]=summary_stat(sums["mixed"][:,pi,column],counts["mixed"][:,pi,column],samples["mixed"][:,pi,column])
        for n in range(1,23):
            r["mixed_fields"][str(n)]=summary_stat(sums["mixed"][:,pi,n-1],counts["mixed"][:,pi,n-1],samples["mixed"][:,pi,n-1])
        r["ppo_minus_net"]=summary_stat(sums["mixed"][:,pi,62],counts["mixed"][:,pi,62],samples["mixed"][:,pi,62])
        r["pool_order"]=sorted(range(1,21),key=lambda n:r["pool_only"][str(n)]["mean"],reverse=True)
        # Average ranks for exact ties avoid arbitrary entrant-ID advantages.
        values=samples["pool"][:,pi,:]
        ranks=1+(values[:,:,None]<values[:,None,:]).sum(axis=2)+.5*((values[:,:,None]==values[:,None,:]).sum(axis=2)-1)
        for n in range(1,21):
            r["pool_only"][str(n)]["bootstrap_rank_interval95"]=[float(x) for x in np.quantile(ranks[:,n-1],[.025,.975])]
            r["pool_only"][str(n)]["bootstrap_top_rank_fraction"]=float(np.mean(ranks[:,n-1]==1))
        result["payouts"][payout]=r
    for group in [totals,*by_cohort.values()]:
        for t in group.values():
            d=t["decisions"]
            for key in ("timeouts","safety_fallbacks","budget_fallbacks","native_fallbacks"):
                t[key+"_fraction"]=t[key]/d if d else 0
            t["mean_latency_us"]=t["total_latency_ns"]/d/1000 if d else 0
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",required=True);p.add_argument("--output",required=True)
    p.add_argument("--pattern",default="production_*.jsonl")
    p.add_argument("--repetitions",type=int,default=4)
    p.add_argument("--bootstrap",type=int,default=5000)
    a=p.parse_args();paths=sorted(Path(a.results).glob(a.pattern))
    if not paths:raise ValueError("no result shards")
    output=analyze(paths,a.repetitions,a.bootstrap)
    Path(a.output).write_text(json.dumps(output,indent=2))
    print(json.dumps({"tables":output["tables"],"orders":{p:r["pool_order"] for p,r in output["payouts"].items()},"faults":0}))

if __name__=="__main__":main()
