"""Estimator sampling error against a much larger same-model sample.

This measures Monte Carlo error, not correctness of inferred opponent ranges.
Uses distinct seeds for runtime and reference; native helper recommended.
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pokerbot.equity import EquityEstimator
from pokerbot.native import LIB

p=argparse.ArgumentParser()
p.add_argument("--positions",type=int,default=300)
p.add_argument("--reference-trials",type=int,default=12000)
p.add_argument("--output",default="benchmarks/equity_accuracy.json")
a=p.parse_args()
rng=random.Random(202611065)
groups={}
worst=[]
for i in range(a.positions):
    n=(1,2,4,7)[i%4]
    street=(3,4,5)[i%3]
    cs=rng.sample(range(52),street+2)
    fractions=[rng.uniform(.1,.8) for _ in range(n)]
    aggression=.7 if i%2 else 0.
    estimate=EquityEstimator(i+717).estimate(cs[:2],cs[2:],fractions,aggression)[0]
    exact=EquityEstimator(i+9711).estimate(cs[:2],cs[2:],fractions,aggression,trials=a.reference_trials)[0]
    error=estimate-exact
    key=f"{n}_opponents"
    groups.setdefault(key,[]).append(error)
    worst.append({"abs_error":abs(error),"estimated":estimate,"reference":exact,"hole":cs[:2],"board":cs[2:],"fractions":fractions})
def summarize(xs):
    return {"n":len(xs),"mean_error":statistics.mean(xs),"mean_absolute_error":statistics.mean(abs(x) for x in xs),"rmse":(statistics.mean(x*x for x in xs))**.5,"max_abs_error":max(abs(x) for x in xs)}
result={"native_accelerator":LIB is not None,"positions":a.positions,"seed":202611065,"reference_trials":a.reference_trials,"scope":"Sampling error relative to same heuristic opponent model, not real poker-model accuracy","all":summarize(sum(groups.values(),[])),"by_opponents":{k:summarize(v) for k,v in groups.items()},"worst_five":sorted(worst,key=lambda r:r["abs_error"],reverse=True)[:5]}
Path(a.output).write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
