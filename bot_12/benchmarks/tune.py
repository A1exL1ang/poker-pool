"""Bounded policy tuning on dedicated local scenario/deal seeds.

This script deliberately does not touch the final held-out seed range.
Outputs include every candidate and paired-reference results, including losses.
Run from workspace: python benchmarks/tune.py --tables 96 --workers 4
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'benchmarks'))
from evaluate import evaluate

BASE = dict(open_scale=1.,call_margin=.025,value_threshold=.70,bluff_scale=1.,
            range_selection=1.,shove_margin=.025,threebet_scale=1.,realization=1.)
VARIANTS = {
    'original': {},
    'open_wider': {'open_scale':1.2},
    'open_tighter': {'open_scale':.85},
    'pressure': {'value_threshold':.65,'bluff_scale':1.5,'range_selection':.8,'call_margin':.015},
    'cautious': {'value_threshold':.75,'bluff_scale':.6,'range_selection':1.2,'call_margin':.04},
    'defend': {'call_margin':.005,'realization':1.08,'range_selection':.8},
    'tight_defend': {'call_margin':.045,'realization':.94},
    'bluff_less': {'bluff_scale':.3},
    'unconditioned': {'range_selection':0.},
    'thin_value': {'value_threshold':.64,'bluff_scale':.5},
    'shove_cautious': {'shove_margin':.15},
    'wide_open_tight_call': {'open_scale':1.35,'call_margin':.04},
}
SCENARIOS = [('mixed','top3','doubling'),('tag','top3','doubling'),
             ('lag','top3','doubling'),('maniac','top3','doubling'),
             ('shove','top3','doubling'),('mixed','top3','gentle'),
             ('selfplay','top3','doubling')]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tables',type=int,default=96,help='per scenario and candidate')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--variants',nargs='*',default=list(VARIANTS))
    p.add_argument('--output',default='measurements/tuning.json')
    a=p.parse_args()
    out=ROOT/a.output
    out.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    report={'split':'tuning','note':'Synthetic scenarios only; independent heldout seed range not accessed.',
            'base':BASE,'tables_per_scenario':a.tables,'scenarios':SCENARIOS,'candidates':{}}
    for name in a.variants:
        params=dict(BASE,**VARIANTS[name])
        rows=[]
        for k,(lineup,payout,schedule) in enumerate(SCENARIOS):
            r=evaluate(tables=a.tables,seed_start=20000+2000*k,lineup=lineup,
                       payout=payout,schedule=schedule,params=params,
                       reference_params=BASE,workers=a.workers,split='training')
            rows.append(r)
        # Equal scenario weights. Risk of overfitting this score is documented.
        mean=sum(r['metrics']['candidate']['points']['mean'] for r in rows)/len(rows)
        delta=sum(r['paired_points_delta']['mean'] for r in rows)/len(rows)
        se=(sum(r['paired_points_delta']['se']**2 for r in rows))**.5/len(rows)
        report['candidates'][name]={'params':params,'mean_points':mean,
             'paired_delta':delta,'paired_se':se,'scenarios':rows}
        report['seconds']=time.time()-started
        out.write_text(json.dumps(report,indent=2))
        print(json.dumps({'variant':name,'mean_points':mean,'paired_delta':delta,
                          'paired_se':se,'seconds':report['seconds']}),flush=True)
    ranked=sorted(report['candidates'],key=lambda n:report['candidates'][n]['paired_delta'],reverse=True)
    print('Ranking: '+', '.join(ranked),flush=True)


if __name__=='__main__':
    main()
