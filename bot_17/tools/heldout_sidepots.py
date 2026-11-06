"""Frozen heterogeneous seven-opponent sidepot fallback accuracy audit."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import equity

def main():
    p=argparse.ArgumentParser();p.add_argument('--spots',type=int,default=600);p.add_argument('--seed',type=int,default=17555191)
    p.add_argument('--output',type=Path,default=ROOT/'data'/'equity_sidepots_heldout.json');args=p.parse_args()
    native=equity._NATIVE
    if native is None or not hasattr(native,'poker_equity_pots'):raise SystemExit('Current native teacher required')
    rng=random.Random(args.seed);errors=[];native_errors=[]
    equity.sampler_stats(reset=True)
    for i in range(args.spots):
        c=rng.sample(range(52),7);nb=rng.choice((3,4,5));ranges=[(rng.random(),rng.random()) for _ in range(7)]
        amounts=[rng.randint(10,1000) for _ in range(7)];pots=[(amounts[n-1],(1<<n)-1) for n in range(1,8)]
        target=equity.estimate(c[:2],c[2:2+nb],ranges=ranges,pots=pots,seed=args.seed+i*971,samples=4096)
        runtime=equity.estimate(c[:2],c[2:2+nb],ranges=ranges,pots=pots,seed=args.seed+i*971+1,samples=64)
        try:
            equity._NATIVE=None
            prediction=equity.estimate(c[:2],c[2:2+nb],ranges=ranges,pots=pots)
        finally:
            equity._NATIVE=native
        errors.append(prediction-target);native_errors.append(runtime-target)
    def summary(values):
        a=sorted(map(abs,values));return {'rmse':math.sqrt(sum(v*v for v in values)/len(values)),
            'mae':sum(a)/len(a),'p95_absolute_error':a[int(len(a)*.95)],'max_absolute_error':max(a)}
    result={'kind':'frozen held-out seven-sidepot audit; no policy tuning','seed':args.seed,'spots':args.spots,
        'teacher_samples':4096,'runtime_native_samples':64,'python_prefix_trees':80,
        'model':summary(errors),'runtime_native':summary(native_errors),'sampler_stats':equity.sampler_stats(),
        'limitations':'Python uses mean range parameters separately within each mask and omits blocker correlations induced by players outside that mask; native samples all ranges jointly. Both share heuristic opponent model.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
