"""Bounded, reproducible strategy experiments. All environments are local assumptions.

Tuning seeds start at 210000, confirmation at 510000; reserved evaluation at
910000. Run tuning before selecting configs; never select using held-out data.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILES = {
    'base': {},
    'cautious': dict(open_scale=.90, value_margin=.065, call_margin=.055, bluff_scale=.5),
    'pressure': dict(open_scale=1.15, value_margin=.015, call_margin=.025, bluff_scale=1.2),
    'call_tight': dict(call_margin=.075),
    'thin_value': dict(value_margin=0.),
    'open_wide': dict(open_scale=1.25),
    'open_tight': dict(open_scale=.80),
    'no_bluff': dict(bluff_scale=0.),
    'no_adapt': dict(adapt=False),
    'no_icm': dict(icm=False),
}
ENVIRONMENTS = [
    ('mixed_top3', 'mixed', 'top3', 'default'),
    ('mixed_winner', 'mixed', 'winner', 'fast'),
    ('tight_top3', 'tight', 'top3', 'default'),
    ('pressure_top3', 'pressure', 'top3', 'default'),
    ('calling_top3', 'calling', 'top3', 'slow'),
    ('selfplay_top3', 'selfplay', 'top3', 'fast'),
]


def one_task(args):
    name, config, env, tables, seed, output, enforce = args
    from agent import Agent
    from tournament import run_benchmark, EXPERIMENTAL_PAYOUTS, LEVELS
    import equity
    label, lineup, payout, levels = env
    def factory(seed):
        return Agent(seed=seed, config=config)
    result = run_benchmark(tables=tables, seed=seed, lineup=lineup,
                           payouts=EXPERIMENTAL_PAYOUTS[payout], levels=LEVELS[levels],
                           hero_factory=factory, enforce_time=enforce, label=name+'_'+label)
    result['config'] = config
    result['backend'] = equity.BACKEND
    result['sampler'] = equity.sampler_stats()
    path = Path(output) / (name + '_' + label + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, separators=(',', ':'))+'\n', encoding='utf-8')
    return name, label, {k:v for k,v in result.items() if k != 'results'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['tune','confirm','heldout'], required=True)
    p.add_argument('--profiles', default='base,cautious,pressure,call_tight,thin_value,open_wide,open_tight,no_bluff,no_adapt,no_icm')
    p.add_argument('--tables', type=int, default=150)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--enforce-time', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.tables <= 10000:
        raise SystemExit('Use 1..8 workers and 1..10000 tables per task')
    output = args.output or ROOT/'measurements'/args.phase
    output.mkdir(parents=True,exist_ok=True)
    profiles = args.profiles.split(',')
    start_seed = {'tune':210000, 'confirm':510000, 'heldout':910000}[args.phase]
    tasks = [(name, PROFILES[name], env, args.tables, start_seed+i*20000,
              str(output), args.enforce_time) for name in profiles for i,env in enumerate(ENVIRONMENTS)]
    summary = {'phase':args.phase, 'tables_per_environment':args.tables,
               'environments':ENVIRONMENTS, 'profiles':{n:PROFILES[n] for n in profiles},
               'seed_start':start_seed, 'workers':args.workers, 'enforce_time':args.enforce_time,
               'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                                for f in ['agent.py','equity.py','risk.py','simulator.py','baselines.py','tournament.py']},
               'results':{}, 'warning':'Synthetic opponents and assumed schedules/payouts; not official contest performance.'}
    begin=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_task,task) for task in tasks]
        for future in as_completed(futures):
            name, env, result = future.result()
            summary['results'].setdefault(name,{})[env]=result
            print(name, env, round(result['hero_score']['mean'],6), 'elapsed',round(time.time()-begin,1), flush=True)
            (output/'summary.partial.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    summary['mean_score_across_environments'] = {
        name:statistics.fmean(summary['results'][name][env[0]]['hero_score']['mean'] for env in ENVIRONMENTS)
        for name in profiles}
    summary['elapsed_seconds']=time.time()-begin
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary['mean_score_across_environments'],indent=2),flush=True)


if __name__=='__main__':
    main()
