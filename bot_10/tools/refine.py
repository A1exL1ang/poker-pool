"""Second-stage development comparisons; these seeds are NOT final evaluation."""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import tournament,summarize
from tools.refinement_policy import Agent as DevelopmentAgent
from tools.experiment_models import load_model

VARIANTS={'baseline':{},'wide':{'open_scale':1.15},'pushfold':{'pushfold':True},
          'pushfold_wide':{'pushfold':True,'open_scale':1.15},
          'pushfold_cautious':{'pushfold':True,'call_margin':.04},
          'pushfold_soft':{'pushfold':True,'range_penalty':.75}}


def evaluate(item):
    name,config,n=item
    rows=[tournament(30000+i,config,'self' if i%5 else 'mixed',
                     'normal' if i%3 else 'slow' if i%2 else 'fast',policy_class=DevelopmentAgent) for i in range(n)]
    return name,{'config':config,'summary':summarize(rows),'rows':rows}


if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 1500;start=time.time()
    workers=min(8,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    model=str(Path('assets/equity_model_8.json').resolve())
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=load_model, initargs=(model,)) as pool:
        results=dict(pool.map(evaluate,[(k,v,n) for k,v in VARIANTS.items()]))
    baseline=results['baseline']['rows']
    for result in results.values():
        result['paired_change']={}
        for metric in ('winner','top3','linear'):
            diff=[r['scores'][metric]-b['scores'][metric] for r,b in zip(result['rows'],baseline)]
            mean=statistics.mean(diff);se=statistics.stdev(diff)/math.sqrt(n)
            result['paired_change'][metric]={'mean':mean,'se':se,'ci95':[mean-1.96*se,mean+1.96*se]}
    Path('results/refinement.json').write_text(json.dumps({'tables_per_variant':n,'seconds':time.time()-start,'results':results},indent=2))
    print(json.dumps({k:{'scores':v['summary']['scores'],'paired_change':v['paired_change']} for k,v in results.items()},indent=2))
