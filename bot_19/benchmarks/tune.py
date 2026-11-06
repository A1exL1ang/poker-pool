"""Bounded policy search; independent validation selects, held-out seeds unused."""
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.run import evaluate, summarize, source_hashes, REFERENCE_PARAMS
from equity_model import asset_status


def variants(count, seed):
    rng=random.Random(seed)
    rows=[dict(REFERENCE_PARAMS)]
    for setting in ({'adapt':False},{'icm':False},{'bluff_scale':0.},
                    {'open_scale':1.25},{'value_threshold':.59},
                    {'range_tightness':1.35},{'call_margin':.06}):
        rows.append({**REFERENCE_PARAMS,**setting})
    for i in range(max(0,count-len(rows))):
        p=dict(REFERENCE_PARAMS)
        p.update(open_scale=rng.choice([.85,1.,1.15,1.30]),
                 call_margin=rng.choice([.005,.025,.045,.065]),
                 value_threshold=rng.choice([.60,.64,.68,.72]),
                 bluff_scale=rng.choice([.25,.65,1.,1.4]),
                 range_tightness=rng.choice([.8,1.,1.2,1.4]),
                 realization=rng.choice([.86,.92,.98]),
                 push_margin=rng.choice([0.,.08,.20]))
        rows.append(p)
    return rows[:count]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidates',type=int,default=28)
    p.add_argument('--games',type=int,default=240)
    p.add_argument('--finalists',type=int,default=6)
    p.add_argument('--validation-games',type=int,default=1200)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seed',type=int,default=192612)
    p.add_argument('--output',default='measurements/tuning.json')
    a=p.parse_args()
    if not all(asset_status()[key] for key in ('postflop_model','preflop_table')):
        p.error('requires trained assets')
    pools=['mixed','tight','pressure','calling','mirror']
    payouts=['chip','top3','linear']
    schedules=['local','fast']
    candidates=variants(a.candidates,a.seed)
    report={'arguments':vars(a),'hashes':source_hashes(),
            'selection_rule':'top tune mean, then independent validation mean; default retained unless validation gain>0.004 and at least3/5 pool means improve',
            'pools':pools,'payouts':payouts,'schedules':schedules,'candidates':[],'validation':[]}
    start=time.perf_counter()
    for i,params in enumerate(candidates):
        rows=evaluate(a.games,'tune',pools,payouts,schedules,params,a.workers)
        entry={'i':i,'params':params,'summary':summarize(rows),
               'by_pool':{pool:summarize([r for r in rows if r['pool']==pool]) for pool in pools},
               'scores':[r['score'] for r in rows]}
        report['candidates'].append(entry)
        print(json.dumps({'phase':'tune','i':i,'mean':entry['summary']['score_mean']}),flush=True)
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    ids=sorted(range(len(candidates)),key=lambda i:report['candidates'][i]['summary']['score_mean'],reverse=True)[:a.finalists]
    if 0 not in ids:
        ids.append(0)
    for i in ids:
        rows=evaluate(a.validation_games,'validation',pools,payouts,schedules,candidates[i],a.workers)
        entry={'i':i,'params':candidates[i],'summary':summarize(rows),
               'by_pool':{pool:summarize([r for r in rows if r['pool']==pool]) for pool in pools},
               'scores':[r['score'] for r in rows]}
        report['validation'].append(entry)
        print(json.dumps({'phase':'validation','i':i,'mean':entry['summary']['score_mean']}),flush=True)
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    base=next(r for r in report['validation'] if r['i']==0)
    chosen=max(report['validation'],key=lambda r:r['summary']['score_mean'])
    gain=chosen['summary']['score_mean']-base['summary']['score_mean']
    improvements=sum(chosen['by_pool'][pool]['score_mean']>base['by_pool'][pool]['score_mean'] for pool in pools)
    if gain<=.004 or improvements<3:
        chosen=base
    diff=[v-b for v,b in zip(chosen['scores'],base['scores'])]
    report['selected']=chosen['i']
    report['selected_params']=chosen['params']
    report['validation_paired_gain']=statistics.mean(diff)
    report['validation_paired_se']=statistics.stdev(diff)/len(diff)**.5 if len(diff)>1 else 0.
    report['elapsed_seconds']=time.perf_counter()-start
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    Path('assets/selected_params.json').write_text(json.dumps(chosen['params'],indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('candidates','validation')},indent=2))


if __name__=='__main__':
    main()
