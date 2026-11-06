"""Bounded tuning/evaluation driver. Run only on local or OWN compute allocation.

All opponents, payouts, blind schedules and seeds are local assumptions. This
does not submit the bot or contact any service. Each process is a simulation
worker, not an agent/CLI session. --workers may not exceed allocated CPUs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.benchmark import run, mean_ci


VARIANTS = {
    'initial': {},
    'cautious': {'call_margin': .055, 'selection': 1.5, 'open_scale': .90, 'aggression': .7},
    'loose': {'call_margin': .005, 'selection': .65, 'open_scale': 1.10},
    'no_bluff': {'aggression': 0.0},
    'more_bluff': {'aggression': 1.6},
    'tight_open': {'open_scale': .8},
    'wide_open': {'open_scale': 1.2},
    'more_selection': {'selection': 1.8},
    'less_selection': {'selection': .5},
    'no_adaptation': {'adaptation': 0.0},
    'more_adaptation': {'adaptation': 2.0},
    'tight_calls': {'call_margin': .07},
    'wide_cautious': {'open_scale':1.2,'selection':1.5,'call_margin':.04},
    'wide_aggressive': {'open_scale':1.2,'selection':1.5,'aggression':1.6},
    'wider_open': {'open_scale':1.4},
    'wide_icm': {'open_scale':1.2,'selection':1.5,'aggression':1.6,'icm_shove':1.0},
}


def task_run(task):
    name, config, suite, payout, tables, seed, enforce, hide, schedule = task
    result = run(kwargs={'config': config}, suite=suite, payout=payout,
                 tables=tables, seed=seed, enforce_timeout=enforce,
                 expose_payouts=not hide, blind_schedule=schedule)
    result['variant'] = name
    result['split'] = 'tuning' if seed < 900000 else 'heldout'
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('tune', 'evaluate'), default='tune')
    p.add_argument('--tables', type=int, default=128)
    p.add_argument('--seed', type=int, default=185000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--suites', default='mixed,calling,tight,aggressive')
    p.add_argument('--payouts', default='top3')
    p.add_argument('--variants', default='all')
    p.add_argument('--config', default='{}')
    p.add_argument('--enforce-timeout', action='store_true')
    p.add_argument('--hide-payouts', action='store_true')
    p.add_argument('--fast-blinds', action='store_true')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error('workers must be 1..8 and fit your own allocation')
    if args.mode == 'evaluate' and args.seed < 900000:
        p.error('reserve seeds >=900000 for final held-out evaluation')
    variants = VARIANTS if args.mode == 'tune' else {'candidate': json.loads(args.config)}
    if args.mode == 'tune':
        # Historical tuning labels keep their original baseline after the
        # selected final defaults change. Every setting is explicit.
        base = {'open_scale':1.0,'call_margin':.025,'aggression':1.0,
                'selection':1.0,'shove_margin':.03,'adaptation':1.0,'icm_shove':0.0}
        variants = {name:{**base,**cfg} for name,cfg in variants.items()}
    if args.variants != 'all' and args.mode == 'tune':
        variants = {k: variants[k] for k in args.variants.split(',')}
    schedule = ((10,20),(20,40),(40,80),(80,160),(160,320),(320,640),(640,1280),(1280,2560)) if args.fast_blinds else None
    tasks = [(name, cfg, suite, pay, args.tables, args.seed + si * 10000 + pi * 100000,
              args.enforce_timeout, args.hide_payouts, schedule)
             for name, cfg in variants.items()
             for si, suite in enumerate(args.suites.split(','))
             for pi, pay in enumerate(args.payouts.split(','))]
    started = time.time()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(task_run, tasks):
            results.append(result)
            print(json.dumps({k: result[k] for k in ('variant','suite','points','overruns','decisions','errors','illegal')}), flush=True)
            output.write_text(json.dumps({'complete':False,'results':results},indent=2),encoding='utf-8')
    summary = {}
    for name in variants:
        runs = [r for r in results if r['variant'] == name]
        summary[name] = {'config':variants[name], 'mean_points':sum(r['points']['mean'] for r in runs)/len(runs),
                         'worst_suite_points':min(r['points']['mean'] for r in runs),
                         'tables':sum(r['tables'] for r in runs),
                         'errors':sum(r['errors'] for r in runs),
                         'illegal':sum(r['illegal'] for r in runs),
                         'overruns':sum(r['overruns'] for r in runs)}
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [ROOT/'agent.py',ROOT/'equity'/'model.json',ROOT/'equity'/'preflop.json'] if p.exists()}
    report = {'complete':True,'mode':args.mode,'arguments':vars(args),'summary':summary,'source_sha256':hashes,
              'elapsed_seconds':time.time()-started,'results':results}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
