"""Locked final protocol. Does not choose or change policy parameters."""
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.run import evaluate,summarize,source_hashes,REFERENCE_PARAMS
from equity_model import asset_status


def paired(a,b):
    diffs=[x['score']-y['score'] for x,y in zip(a,b)]
    mean=statistics.mean(diffs)
    se=statistics.stdev(diffs)/math.sqrt(len(diffs))
    return {'tables':len(diffs),'mean_gain':mean,'paired_se':se,
            'paired_95ci_normal':[mean-1.96*se,mean+1.96*se]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--games',type=int,default=5400)
    p.add_argument('--baseline-games',type=int,default=1800)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--output',default='measurements/final_tournaments.json')
    a=p.parse_args()
    params=json.loads((ROOT/'assets/selected_params.json').read_text())
    pools=['mixed','tight','pressure','calling','mirror','diverse']
    payouts=['chip','top3','linear','top2','satellite']
    schedules=['local','fast','slow']
    report={'kind':'locked held-out local tournament test; not contest performance',
            'arguments':vars(a),'selected_params':params,'reference_params':REFERENCE_PARAMS,
            'heldout_seed_start':19900000,'pools':pools,'payouts':payouts,'schedules':schedules,
            'new_sensitivity_cases':['diverse opponents','top2 points','satellite points','slow blinds'],
            'selection_performed_here':False,'hashes':source_hashes(),'asset_status':asset_status(),
            'python':platform.python_version(),'platform':platform.platform(),'candidates':{}}
    begin=time.perf_counter()
    stored={}
    for label,par,candidate,games in [('selected',params,'agent',a.games),
                                      ('reference',REFERENCE_PARAMS,'agent',a.games),
                                      ('static_tag',None,'tag',a.baseline_games)]:
        rows=evaluate(games,'heldout',pools,payouts,schedules,par,a.workers,candidate=candidate)
        stored[label]=rows
        report['candidates'][label]={'summary':summarize(rows),
            'by_pool':{v:summarize([r for r in rows if r['pool']==v]) for v in pools},
            'by_payout':{v:summarize([r for r in rows if r['payout']==v]) for v in payouts},
            'by_schedule':{v:summarize([r for r in rows if r['schedule']==v]) for v in schedules},
            'rows':rows}
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'candidate':label,'summary':report['candidates'][label]['summary']}),flush=True)
    report['paired_selected_minus_reference']=paired(stored['selected'],stored['reference'])
    report['paired_by_pool']={pool:paired([r for r in stored['selected'] if r['pool']==pool],
                                        [r for r in stored['reference'] if r['pool']==pool]) for pool in pools}
    report['elapsed_seconds']=time.perf_counter()-begin
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['paired_selected_minus_reference']),flush=True)


if __name__=='__main__':
    main()
