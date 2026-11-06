"""Paired-seed development comparison against a frozen own-policy reference."""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import tournament, summarize
from tools.reference_policy import Agent as DevelopmentAgent
from tools.experiment_models import load_model

VARIANTS={'baseline':{},'open_wide':{'open_scale':1.22},'open_tight':{'open_scale':.8},
          'call_loose':{'call_margin':0.},'call_tight':{'call_margin':.06},
          'range_soft':{'range_penalty':.55},'range_hard':{'range_penalty':1.5},
          'no_bluff':{'bluff_scale':0.},'more_bluff':{'bluff_scale':1.8},
          'aggressive':{'aggression':1.25}}


def evaluate(item):
    name,config,n=item
    rows=[]
    for i in range(n):
        lineup='self' if i%5 else 'mixed'
        schedule='normal' if i%3 else 'slow' if i%2 else 'fast'
        rows.append(tournament(20000+i,config,lineup,schedule,policy_class=DevelopmentAgent))
    return name, {'config':config,'summary':summarize(rows),'rows':rows}


if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 500
    start=time.time()
    model=str(Path('assets/equity_model_24.json').resolve())
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(8,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))), initializer=load_model, initargs=(model,)) as pool:
        results=dict(pool.map(evaluate,[(name,config,n) for name,config in VARIANTS.items()]))
    baseline=results['baseline']['rows']
    for result in results.values():
        result['paired_change']={}
        for metric in ('winner','top3','linear'):
            diff=[r['scores'][metric]-b['scores'][metric] for r,b in zip(result['rows'],baseline)]
            mean=statistics.mean(diff); se=statistics.stdev(diff)/math.sqrt(n)
            result['paired_change'][metric]={'mean':mean,'se':se,'ci95':[mean-1.96*se,mean+1.96*se]}
    Path('results/tuning.json').write_text(json.dumps({'tables_per_variant':n,'seconds':time.time()-start,'results':results},indent=2))
    print(json.dumps({k:{'score':v['summary']['scores'],'paired_change':v['paired_change']} for k,v in results.items()},indent=2))
