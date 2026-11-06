"""Bounded local-policy selection. TUNING seeds only, not final evidence.

Uses independent synthetic opponent families and supplied synthetic payouts.
Process workers are ordinary offline simulations, never agents or online calls.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.benchmark import run_benchmark, summary

CANDIDATES = {
    'base': {},
    'conditional_off': {'conditional_weight': 0.},
    'tight_call': {'call_margin': .06, 'conditional_weight': .95, 'bluff_prior': .10, 'bet_penalty_scale': 1.25},
    'loose_call': {'call_margin': 0., 'conditional_weight': .55, 'bluff_prior': .30, 'bet_penalty_scale': .65},
    'aggressive': {'open_scale': 1.2, 'cbet': .70, 'bluff_scale': 1.8, 'value_threshold': .65},
    'cautious': {'open_scale': .85, 'call_margin': .05, 'value_threshold': .75, 'cbet': .4, 'bluff_scale': .5},
    'icm_off': {'risk_scale': 0.},
    'thin_value': {'value_threshold': .62, 'bluff_scale': .5, 'cbet': .35},
    'low_bluff': {'bluff_scale': 0., 'conditional_weight': .85, 'cbet': .4, 'open_scale': 1.05},
}
SUITES = [
    ('mixed_sng', 'mixed', 'sng', 1000, 1.),
    ('mixed_wta', 'mixed', 'wta', 2000, 1.),
    ('mirror_sng', 'mirror', 'sng', 3000, 1.),
    ('nit_sng', 'nit', 'sng', 4000, 1.),
    ('mc_mixed_sng', 'mc,value,nit,loose,station,mirror,value', 'sng', 5000, .25),
]


def task(config):
    name, params, suite, count, model_path = config
    label, lineup, payout, seed, scale = suite
    args = SimpleNamespace(tables=max(8,int(count*scale)),max_hands=200,seed_start=seed,
        split='tuning',lineup=lineup,params=json.dumps(params),blinds=None,payouts_json=None,
        payouts=payout,model_path=model_path,initial_stack=2000,enforce_time_limit=False,
        no_timing=True,progress=0,stress=False)
    result = run_benchmark(args)
    return name, label, result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tables',type=int,default=96)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--candidates',help='JSON candidate name->parameter object')
    p.add_argument('--suites',help='JSON suite tuples [label,lineup,payout,seed,scale]')
    p.add_argument('--model-path')
    p.add_argument('--out',default='measurements/tuning_selection.json')
    a=p.parse_args()
    candidates=json.loads(Path(a.candidates).read_text()) if a.candidates else CANDIDATES
    suites=json.loads(Path(a.suites).read_text()) if a.suites else SUITES
    todo=[(name,params,suite,a.tables,a.model_path) for name,params in candidates.items() for suite in suites]
    results={name:{} for name in candidates}
    started=time.time()
    with ProcessPoolExecutor(max_workers=min(8,a.workers)) as pool:
        for name,label,result in pool.map(task,todo):
            results[name][label]=result
            print(f'{name} {label}: {result["focal_points"]["mean"]:.6f}',flush=True)
    scores={name:statistics.mean(r['focal_points']['mean']/sum(r['config']['payouts']) for r in suites_result.values()) for name,suites_result in results.items()}
    paired={}
    base=results.get('base')
    if base:
        for name,group in results.items():
            paired[name]={}
            for label,r in group.items():
                d=[x['points']-y['points'] for x,y in zip(r['tables'],base[label]['tables'])]
                paired[name][label]=summary(d)
    report={'purpose':'policy selection only; no held-out tournament evidence',
            'candidates':candidates,'suite_definitions':suites,'scores':scores,
            'ranking':sorted(scores,key=scores.get,reverse=True),'paired_difference_vs_base':paired,
            'elapsed_seconds':time.time()-started,'results':results}
    Path(a.out).parent.mkdir(exist_ok=True,parents=True)
    Path(a.out).write_text(json.dumps(report,indent=2))
    print(json.dumps({'scores':scores,'ranking':report['ranking'],'elapsed_seconds':report['elapsed_seconds']},indent=2))


if __name__=='__main__':
    main()
