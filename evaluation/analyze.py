"""Audit the full schedule, then bootstrap whole seed blocks (not individual seats)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def summarize(values, indices):
    values=np.asarray(values,dtype=float)
    samples=values[indices].mean(axis=1)
    return {'mean':float(values.mean()),'ci95':[float(x) for x in np.quantile(samples,[.025,.975])], 'seed_blocks':len(values)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',default='evaluation/results');ap.add_argument('--seeds',type=int,default=64);ap.add_argument('--output',default='evaluation/summary.json');a=ap.parse_args()
    paths=sorted(Path(a.results).glob('production_*.jsonl'))
    rows=[json.loads(l) for p in paths for l in p.read_text().splitlines()]
    expected={f'{p}:{s}:{o}:{r}' for p in ('top3','winner','top4') for s in range(a.seeds) for o in range(9) for r in range(8)}
    actual=[r['match_id'] for r in rows]
    assert len(actual)==len(set(actual)), 'duplicate matches'
    assert set(actual)==expected, f'missing={len(expected-set(actual))}, extra={len(set(actual)-expected)}'
    for r in rows:
        assert len(set(r['seats']))==8 and set(r['seats'])==set(range(9))-{r['omitted']}
        assert sum(r['final_stacks'])==16000 and min(r['final_stacks'])>=0
        assert abs(sum(r['points'])-1)<1e-8 and min(r['points'])>=0
        assert not any(r['errors']+r['illegal']+r['internal_errors']),f"fault in {r['match_id']}"
    rng=np.random.default_rng(20261106)
    indices=rng.integers(0,a.seeds,(20000,a.seeds))
    output={'tables':len(rows),'seed_blocks':a.seeds,'bootstrap_replicates':len(indices),'method':'Percentile bootstrap of complete seed blocks, including all seat rotations and relevant fields; intervals are marginal, not multiplicity-adjusted.', 'source_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'payouts':{},'timing':{}}
    for payout in ('top3','winner','top4'):
        pr=[r for r in rows if r['payout']==payout]
        result={'eight_only':{},'mixed_fields':{},'paired_vs_claude':{}}
        for bot in range(9):
            for label,subset in [('eight_only',[r for r in pr if r['omitted']==0 and bot in r['seats']]),('mixed_fields',[r for r in pr if bot in r['seats']])]:
                if not subset:continue
                vals=[np.mean([r['points'][r['seats'].index(bot)] for r in subset if r['seed_index']==s]) for s in range(a.seeds)]
                result[label][str(bot)]=dict(summarize(vals,indices),tables=len(subset))
            if bot:
                shared=[r for r in pr if bot in r['seats'] and 0 in r['seats']]
                vals=[np.mean([r['points'][r['seats'].index(bot)]-r['points'][r['seats'].index(0)] for r in shared if r['seed_index']==s]) for s in range(a.seeds)]
                result['paired_vs_claude'][str(bot)]=dict(summarize(vals,indices),tables=len(shared))
        result['eight_only_order']=sorted(range(1,9),key=lambda b:result['eight_only'][str(b)]['mean'],reverse=True)
        output['payouts'][payout]=result
    for bot in range(9):
        output['timing'][str(bot)]={k:sum(r[k][r['seats'].index(bot)] for r in rows if bot in r['seats']) for k in ('decisions','timeouts','errors','illegal','internal_errors','total_latency_ns')}
        t=output['timing'][str(bot)];t['timeout_fraction']=t['timeouts']/t['decisions'];t['mean_us']=t['total_latency_ns']/t['decisions']/1000
    Path(a.output).write_text(json.dumps(output,indent=2))
    print(json.dumps({'tables':len(rows),'orders':{p:r['eight_only_order'] for p,r in output['payouts'].items()},'faults':0}))

if __name__=='__main__':main()
