"""Frozen-candidate paired evaluation on seeds never used in policy tuning."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import tournament,summarize

# Different tables in each cohort and from seeds20000/30000 used for tuning.
COHORTS=[('self_normal','self','normal',2000),('self_slow','self','slow',1000),
         ('self_fast','self','fast',1000),('challenge_normal','challenge','normal',2000),
         ('mixed_normal','mixed','normal',1000),('loose_normal','loose','normal',1000),
         ('tight_normal','tight','normal',1000),('challenge_fast','challenge','fast',1000)]


def cohort(item):
    index,(name,lineup,schedule,n)=item
    seed=60000+index*100000
    candidate=[];reference=[]
    for i in range(n):
        candidate.append(tournament(seed+i,lineup=lineup,schedule=schedule))
        reference.append(tournament(seed+i,lineup=lineup,schedule=schedule,reference_hero=True))
    return name,{'lineup':lineup,'schedule':schedule,'seed_start':seed,'tables':n,
                 'candidate':summarize(candidate),'reference':summarize(reference),
                 'paired_change':paired(candidate,reference),'candidate_rows':candidate,'reference_rows':reference}


def paired(candidate,reference):
    report={}
    for metric in ('winner','top3','linear'):
        diff=[a['scores'][metric]-b['scores'][metric] for a,b in zip(candidate,reference)]
        mean=statistics.mean(diff);se=statistics.stdev(diff)/math.sqrt(len(diff))
        report[metric]={'mean':mean,'se':se,'ci95':[mean-1.96*se,mean+1.96*se]}
    return report


if __name__=='__main__':
    start=time.time();workers=min(8,int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        results=dict(pool.map(cohort,enumerate(COHORTS)))
    candidate=[r for x in results.values() for r in x['candidate_rows']]
    reference=[r for x in results.values() for r in x['reference_rows']]
    files=['agent.py','equity.py','features.py','pushfold.py','assets/equity_model.json','assets/preflop_equity.json','assets/pushfold.json']
    report={'frozen_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
            'seconds':time.time()-start,'candidate':summarize(candidate),'reference':summarize(reference),
            'paired_change':paired(candidate,reference),'cohorts':results,
            'note':'Held-out seeds and challenge policy family. Synthetic opponents/payouts/schedules; not evidence of official contest win probability. Confidence intervals are approximate normal intervals across independent tables; paired comparisons use shared deal seeds.'}
    Path('results/final_evaluation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cohorts'},indent=2))
