"""Reproducible local tournament evaluation with independent seed ranges.

Example: python benchmarks/run.py --games 1000 --split heldout --workers 4
All payout/schedule choices are local sensitivity assumptions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import Agent, DEFAULTS
from baselines import Baseline, POOLS, make_baseline
from equity_model import asset_status
from sim import Tournament, LOCAL_BLIND_SCHEDULE

PAYOUTS = {
    'chip': None,
    'winner': [1., 0., 0., 0., 0., 0., 0., 0.],
    'top3': [.5, .3, .2, 0., 0., 0., 0., 0.],
    'linear': [8/36, 7/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36],
    'top2': [.65,.35,0.,0.,0.,0.,0.,0.],
    'satellite': [.5,.5,0.,0.,0.,0.,0.,0.],
}
SCHEDULES = {
    'local': LOCAL_BLIND_SCHEDULE,
    'fast': ((10,20),(20,40),(40,80),(80,160),(160,320),(320,640),(640,1280),(1280,2560)),
    'slow': ((10,20),(12,25),(15,30),(20,40),(30,60),(40,80),(60,120),(80,160)),
}
SPLITS = {'tune': 1910000, 'validation': 1950000, 'heldout': 19900000}
# Freeze the opponent policy parameters before candidate selection. This keeps
# mirror comparisons stable if the final Agent defaults change after tuning.
REFERENCE_PARAMS = {
    'open_scale': 1.0, 'call_margin': .025, 'value_threshold': .66,
    'bluff_scale': 1.0, 'range_tightness': 1.0, 'realization': .92,
    'push_margin': .08, 'adapt': True, 'icm': True,
}


def game(task):
    index, seed, pool, payout, schedule, params, enforce, candidate = task
    rng = random.Random(seed)
    hero = index % 8
    styles = list(POOLS[pool])
    rng.shuffle(styles)
    agents = []
    for i in range(8):
        if i == hero:
            a = Agent(seed=seed*11+i, params=params) if candidate == 'agent' else Baseline(candidate, seed*11+i)
        else:
            name = styles.pop()
            a = Agent(seed=seed*13+i, params=REFERENCE_PARAMS) if name == 'mirror' else make_baseline(name, seed*13+i)
        agents.append(a)
    r = Tournament(agents, seed=seed, payouts=PAYOUTS[payout], blind_schedule=SCHEDULES[schedule],
                   enforce_time_limit=enforce).run()
    times = r.stats.latencies_ns[hero]
    score = r.chip_shares[hero] if r.scores is None else r.scores[hero]
    return {'i': index, 'seed': seed, 'hero': hero, 'pool': pool, 'payout': payout, 'schedule': schedule,
            'score': score, 'chip_share': r.chip_shares[hero], 'alive': r.stacks[hero]>0,
            'champion': r.champion == hero, 'hands': r.hands, 'capped': r.hands == 200 and r.champion is None,
            'decisions': r.stats.decisions[hero], 'invalids': r.stats.invalids[hero],
            'errors': r.stats.errors[hero], 'overruns': r.stats.overruns[hero],
            'fallbacks': getattr(agents[hero], 'fallbacks', 0),
            'latency_sum_ns': sum(times), 'latency_max_ns': max(times, default=0),
            'latency_sample_ns': times[::max(1, len(times)//50)]}


def summarize(rows):
    vals = [r['score'] for r in rows]
    n = len(vals)
    mean = statistics.mean(vals) if n else 0.
    se = statistics.stdev(vals)/math.sqrt(n) if n>1 else 0.
    times = sorted(t for r in rows for t in r['latency_sample_ns'])
    decisions = sum(r['decisions'] for r in rows)
    return {'tables': n, 'score_mean': mean, 'score_se': se,
            'score_95ci_normal': [mean-1.96*se, mean+1.96*se], 'equal_seat_reference': .125,
            'chip_share_mean': statistics.mean(r['chip_share'] for r in rows) if rows else 0,
            'survival_rate': sum(r['alive'] for r in rows)/max(1,n),
            'outright_win_rate': sum(r['champion'] for r in rows)/max(1,n),
            'cap_rate': sum(r['capped'] for r in rows)/max(1,n),
            **{key: sum(r[key] for r in rows) for key in ('decisions','invalids','errors','overruns','fallbacks')},
            'latency_us': {'mean': sum(r['latency_sum_ns'] for r in rows)/max(1,decisions)/1000,
                           'p50_sample': times[len(times)//2]/1000 if times else 0,
                           'p99_sample': times[min(len(times)-1,int(len(times)*.99))]/1000 if times else 0,
                           'max': max((r['latency_max_ns'] for r in rows),default=0)/1000}}


def source_hashes():
    files = ['agent.py','icm.py','sim.py','poker_eval.py','equity_model.py','equity_features.py','baselines.py',
             'assets/preflop.json','assets/equity_model.json','assets/equity_fast.py',
             'benchmarks/run.py','benchmarks/tune.py','benchmarks/latency.py']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files if (ROOT/p).exists()}


def evaluate(games, split, pools, payouts, schedules, params, workers=1, enforce=False, candidate='agent', offset=0):
    tasks = [(i, SPLITS[split]+i+offset, pools[i%len(pools)], payouts[(i//len(pools))%len(payouts)],
              schedules[(i//(len(pools)*len(payouts)))%len(schedules)], params, enforce, candidate) for i in range(games)]
    if workers == 1:
        rows = [game(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(8,workers)) as exe:
            rows = list(exe.map(game, tasks, chunksize=4))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games',type=int,default=500)
    p.add_argument('--split',choices=SPLITS,default='validation')
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--pools',default='mixed,tight,pressure,calling,mirror')
    p.add_argument('--payouts',default='chip,top3,linear')
    p.add_argument('--schedules',default='local')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--params',default=None)
    p.add_argument('--candidate',default='agent')
    p.add_argument('--enforce',action='store_true')
    p.add_argument('--allow-fallback',action='store_true')
    p.add_argument('--output',default='measurements/tournaments.json')
    a=p.parse_args()
    status=asset_status()
    if not a.allow_fallback and (not status['postflop_model'] or not status['preflop_table']):
        p.error('trained assets missing; only development runs may use --allow-fallback')
    params=json.loads(Path(a.params).read_text()) if a.params else None
    start=time.perf_counter()
    rows=evaluate(a.games,a.split,a.pools.split(','),a.payouts.split(','),a.schedules.split(','),params,a.workers,a.enforce,a.candidate,a.offset)
    report={'kind':'local synthetic tournament evaluation; not contest score', 'arguments':vars(a),
            'parameters':params or DEFAULTS,'asset_status':status,'hashes':source_hashes(),
            'python':platform.python_version(),'platform':platform.platform(),
            'elapsed_seconds':time.perf_counter()-start,'summary':summarize(rows),
            'by_pool':{pool:summarize([r for r in rows if r['pool']==pool]) for pool in a.pools.split(',')},
            'by_payout':{pay:summarize([r for r in rows if r['payout']==pay]) for pay in a.payouts.split(',')},
            'by_schedule':{sch:summarize([r for r in rows if r['schedule']==sch]) for sch in a.schedules.split(',')},
            'rows':rows}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
