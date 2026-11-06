"""Frozen-policy final evaluation on untouched seeds and additional scenarios.

No selection or tuning is performed by this script. Reports contain every
predeclared scenario, including losses, plus paired original-policy references.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'benchmarks'))
from evaluate import evaluate,SCENARIOS
from tune import BASE

CASES=dict(SCENARIOS,fast=('mixed','top3','fast'),
           linear=('mixed','linear','doubling'),top4=('mixed','top4','doubling'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tables',type=int,default=1024)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--params',required=True)
    p.add_argument('--tag-audit',action='store_true')
    p.add_argument('--output',default='measurements/heldout.json')
    a=p.parse_args()
    params=json.loads((ROOT/a.params).read_text())
    params=params.get('params',params)
    out=ROOT/a.output
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'split':'heldout','seed_start':1000000,'scenario_seed_stride':10000,
            'tables_per_scenario':a.tables,'frozen_params':params,'cases':{},
            'note':'Final evaluation only; no policy or model choice from these results. Synthetic scenarios, not official contest payouts/schedule/opponents.'}
    started=time.time()
    for k,(name,(lineup,payout,schedule)) in enumerate(CASES.items()):
        r=evaluate(tables=a.tables,seed_start=1000000+10000*k,lineup=lineup,
                   payout=payout,schedule=schedule,params=params,
                   reference_params=BASE,workers=a.workers,split='heldout')
        report['cases'][name]=r
        report['seconds']=time.time()-started
        out.write_text(json.dumps(report,indent=2))
        print(json.dumps({'scenario':name,'points':r['metrics']['candidate']['points'],
              'paired_original_delta':r['paired_points_delta'],'seconds':report['seconds']}),flush=True)
    if a.tag_audit:
        report['tag_audit']={}
        for k,name in enumerate(('mixed','selfplay','varied','winner')):
            lineup,payout,schedule=CASES[name]
            r=evaluate(tables=a.tables,seed_start=2000000+10000*k,lineup=lineup,
                       payout=payout,schedule=schedule,params=params,
                       reference_style='tag',workers=a.workers,split='heldout')
            report['tag_audit'][name]=r
            report['seconds']=time.time()-started
            out.write_text(json.dumps(report,indent=2))
            print(json.dumps({'tag_audit':name,'points':r['metrics']['candidate']['points'],
                  'paired_tag_delta':r['paired_points_delta'],'seconds':report['seconds']}),flush=True)
    n=len(report['cases'])
    report['equal_scenario_mean_points']=sum(r['metrics']['candidate']['points']['mean'] for r in report['cases'].values())/n
    report['equal_scenario_original_delta']=sum(r['paired_points_delta']['mean'] for r in report['cases'].values())/n
    report['equal_scenario_original_delta_se']=sum(r['paired_points_delta']['se']**2 for r in report['cases'].values())**.5/n
    report['seconds']=time.time()-started
    out.write_text(json.dumps(report,indent=2))
    print('BOT_12_HELDOUT_COMPLETE',flush=True)


if __name__=='__main__':
    main()
