"""Fresh-seed paired policy validation; selection data, not final held-out data."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'benchmarks'))
from evaluate import evaluate
from tune import BASE,SCENARIOS,VARIANTS

VARIANTS=dict(VARIANTS,balanced={'open_scale':1.15,'call_margin':.04,
                'value_threshold':.73,'bluff_scale':.6,'range_selection':1.1,
                'realization':.97})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tables',type=int,default=512)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--variants',nargs='+',default=['wide_open_tight_call','cautious','tight_defend','balanced'])
    p.add_argument('--output',default='measurements/validation.json')
    a=p.parse_args()
    out=ROOT/a.output
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'split':'validation','seed_start':80000,'scenario_seed_stride':5000,
            'tables_per_scenario':a.tables,'note':'Used for policy selection. Heldout seeds>=1000000 not accessed.',
            'base':BASE,'candidates':{}}
    started=time.time()
    for name in a.variants:
        rows=[]
        params=dict(BASE,**VARIANTS[name])
        for k,(lineup,payout,schedule) in enumerate(SCENARIOS):
            r=evaluate(tables=a.tables,seed_start=80000+5000*k,lineup=lineup,
                       payout=payout,schedule=schedule,params=params,
                       reference_params=BASE,workers=a.workers,split='validation')
            rows.append(r)
        mean=sum(r['metrics']['candidate']['points']['mean'] for r in rows)/len(rows)
        delta=sum(r['paired_points_delta']['mean'] for r in rows)/len(rows)
        se=sum(r['paired_points_delta']['se']**2 for r in rows)**.5/len(rows)
        report['candidates'][name]={'params':params,'mean_points':mean,
             'paired_delta':delta,'paired_se':se,'scenarios':rows}
        report['seconds']=time.time()-started
        out.write_text(json.dumps(report,indent=2))
        print(json.dumps({'variant':name,'mean_points':mean,'paired_delta':delta,
                          'paired_se':se,'seconds':report['seconds']}),flush=True)


if __name__=='__main__':
    main()
