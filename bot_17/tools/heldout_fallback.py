"""Final frozen-model equity comparison on separately seeded boards.

This measures agreement with our own range model, not tournament strength.
Do not use this seed for model or policy tuning.
"""
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
    p=argparse.ArgumentParser();p.add_argument('--spots',type=int,default=4000);p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--seed',type=int,default=17555091)
    p.add_argument('--output',type=Path,default=ROOT/'data'/'equity_model_heldout.json')
    args=p.parse_args()
    if equity._NATIVE is None:raise SystemExit('Native teacher required')
    rng=random.Random(args.seed);model_errors=[];heuristic_errors=[];mc_errors=[];groups={}
    equity.sampler_stats(reset=True)
    for i in range(args.spots):
        c=rng.sample(range(52),7);nb=(3,4,5)[rng.randrange(3)];choice=rng.randrange(10)
        opponents=1 if choice<4 else 2 if choice<6 else rng.randrange(3,8)
        tight=rng.random();aggression=rng.random()
        if i%7==0:tight=0.
        if i%11==0:tight=1.
        if i%6==0:aggression=0.
        if i%13==0:aggression=1.
        h,b=c[:2],c[2:2+nb]
        target=equity.estimate(h,b,opponents,tight,args.seed+i*397,args.samples,aggression)
        model=equity.model_estimate(h,b,opponents,tight,aggression)
        heuristic=equity._heuristic(h,b,opponents,tight,aggression)
        mc=equity.estimate(h,b,opponents,tight,args.seed+i*397+1,96 if opponents<4 else 64,aggression)
        model_errors.append(model-target);heuristic_errors.append(heuristic-target);mc_errors.append(mc-target)
        key=f'board{nb}_opponents{opponents}'
        g=groups.setdefault(key,{'n':0,'model_squared_error':0.,'heuristic_squared_error':0.,'runtime_native_squared_error':0.})
        g['n']+=1;g['model_squared_error']+=(model-target)**2;g['heuristic_squared_error']+=(heuristic-target)**2;g['runtime_native_squared_error']+=(mc-target)**2
    def summary(errors):
        absolute=sorted(map(abs,errors));return {'rmse':math.sqrt(sum(x*x for x in errors)/len(errors)),
            'mae':sum(absolute)/len(errors),'p95_absolute_error':absolute[int(len(errors)*.95)],'max_absolute_error':max(absolute)}
    result={'kind':'held-out equity estimator evaluation, not tournament performance','seed':args.seed,'spots':args.spots,
        'teacher_samples':args.samples,'teacher_river_heads_up':'exact enumeration',
        'model_training_rows':equity._MODEL['train_rows'],'model_trees':len(equity._MODEL['trees']),
        'model':summary(model_errors),'heuristic':summary(heuristic_errors),'runtime_native':summary(mc_errors),
        'groups':groups,'sampler_stats':equity.sampler_stats(),
        'limitations':'Teacher and model share heuristic range assumptions; no claim of game-theoretic optimality or calibrated opponent beliefs.'}
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))

if __name__=='__main__':main()
