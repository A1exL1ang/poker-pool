"""Reproducible local tournament evaluation, with explicit synthetic payouts.

Development seeds (1104000+) and final held-out seeds (1190000+) are separate.
Results describe this independent simulator and simple opponents only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from pokerbot.agent import Agent
from pokerbot.baselines import make_baseline
from pokerbot.engine import Tournament

ECOSYSTEMS = {
    'mixed': ['tag', 'lag', 'station', 'tight_passive', 'pushfold', 'tag', 'random'],
    'tight': ['tag', 'tag', 'tight_passive', 'pushfold', 'tag', 'tight_passive', 'tag'],
    'loose': ['lag', 'station', 'random', 'lag', 'station', 'tag', 'lag'],
    'selfplay': ['self'] * 6 + ['tag'],
}
PAYOUTS = {'none': None, 'winner': [1., 0., 0., 0., 0., 0., 0., 0.],
           'top3': [.5, .3, .2, 0., 0., 0., 0., 0.],
           'linear': [7/28, 6/28, 5/28, 4/28, 3/28, 2/28, 1/28, 0.]}


def one(task):
    seed, ecosystem, config, payout_name, enforce, blind = task
    hero = seed % 8
    agents = []
    styles = ECOSYSTEMS[ecosystem]
    j = 0
    candidate = Agent(seed=seed ^ 110011, config=config)
    for seat in range(8):
        if seat == hero:
            agents.append(candidate)
        else:
            style = styles[(j + seed//8) % 7]
            agents.append(Agent(seed=seed * 17 + seat) if style == 'self' else make_baseline(style, seed=seed * 17 + seat))
            j += 1
    payouts = PAYOUTS[payout_name]
    schedule = [(10*2**i, 20*2**i) for i in range(8)] if blind == 'double' else [(max(1, int(10*1.5**i)), max(2, int(20*1.5**i))) for i in range(8)]
    result = Tournament(agents, seed=seed, blind_schedule=schedule, payouts=payouts, enforce_time_limit=enforce).run()
    return {'seed': seed, 'hero_seat': hero, 'chip_share': result['stacks'][hero] / 16000,
            'points': result['points'][hero] if payouts else None,
            'rank': result['expected_ranks'][hero], 'hands': result['hands'],
            'engine_errors': result['errors'], 'invalid_actions': result['invalid_actions'],
            'timeouts': result['timeouts'], 'agent_errors': candidate.diagnostics['errors'],
            'max_decision_ns': result['max_decision_ns']}


def summarize(rows, key):
    values = [r[key] for r in rows if r[key] is not None]
    mean = statistics.mean(values)
    stderr = statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else 0.
    return {'mean': mean, 'stderr': stderr, 'normal_95ci': [mean-1.96*stderr, mean+1.96*stderr]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tables', type=int, default=500)
    p.add_argument('--seed', type=int, default=1104000)
    p.add_argument('--ecosystem', choices=ECOSYSTEMS, default='mixed')
    p.add_argument('--payouts', choices=PAYOUTS, default='none')
    p.add_argument('--blinds', choices=['double', '1.5x'], default='double')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--config', default='{}', help='JSON overrides to policy parameters')
    p.add_argument('--enforce', action='store_true', help='Apply local 1ms wall-clock fallback')
    p.add_argument('--out', default='benchmarks/development.json')
    args = p.parse_args()
    config = json.loads(args.config)
    tasks = [(args.seed+i, args.ecosystem, config, args.payouts, args.enforce, args.blinds) for i in range(args.tables)]
    start = time.monotonic()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(one, tasks, chunksize=8))
    else:
        rows = [one(t) for t in tasks]
    report = {'assumptions': 'Independent local standard NLHE engine; synthetic payouts, if supplied, are hypotheses, not official; configurable assumed blind growth.',
              'arguments': vars(args), 'config': config, 'tables': len(rows), 'seconds': time.monotonic()-start,
              'chip_share': summarize(rows, 'chip_share'), 'expected_rank': summarize(rows, 'rank'),
              'totals': {k:sum(r[k] for r in rows) for k in ('engine_errors','invalid_actions','timeouts','agent_errors','hands')},
              'rows': rows}
    if args.payouts != 'none':
        report['synthetic_points'] = summarize(rows, 'points')
    report['sources'] = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path('pokerbot/agent.py'), Path('assets/equity.bin.z'), Path('assets/preflop.json'))}
    Path(args.out).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','sources')}, indent=2))


if __name__ == '__main__':
    main()
