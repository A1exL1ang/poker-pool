"""Predeclared final evaluation scenarios; no policy selection on these results.

Compare the frozen tuned bot with both a simple baseline and the untuned bot,
using paired per-hand seeds and independent held-out opponent policy families.
All payouts and the unknown blind schedule are explicitly local assumptions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from bot.agent import Agent, Config
from benchmarks.challenge import EquityBaseline, PolarizedBettor
from benchmarks.engine import play_table
from benchmarks.opponents import BaselineAgent
from benchmarks.tournament import POPULATIONS, summary, candidate_seat


PAYOUTS = {
    'winner': [1, 0, 0, 0, 0, 0, 0, 0],
    'top3': [.5, .3, .2, 0, 0, 0, 0, 0],
    'flat3': [.34, .33, .33, 0, 0, 0, 0, 0],
    'linear': [8/36, 7/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36],
}


def opponents(kind, seed):
    if kind == 'challenge':
        return [EquityBaseline(s, seed * 13 + i) for i, s in enumerate(('tight', 'balanced', 'loose'))] + [
            PolarizedBettor(seed * 13 + 3, .05, .8),
            PolarizedBettor(seed * 13 + 4, .22, 1.25),
            PolarizedBettor(seed * 13 + 5, .45, 1.5),
            EquityBaseline('balanced', seed * 13 + 6)]
    if kind == 'mirror':
        return [Agent(seed * 13 + i, Config()) for i in range(7)]
    return [BaselineAgent(style, seed * 13 + i) for i, style in enumerate(POPULATIONS[kind])]


def task(args):
    scenario, config, tables, start, policy, enforce = args
    kind, payout, schedule, scenario_id = scenario
    rows = []
    candidate_counts = dict(decisions=0, over_1ms=0, internal_fallbacks=0)
    for k in range(tables):
        seed = 8100001 + scenario_id * 10000 + start + k
        # The engine's initial button is seed%8. Advancing seat with seed would
        # accidentally keep a constant relative starting position. Cross all
        # 64 seat/button pairs once per64tables instead.
        seat = candidate_seat(start + k)
        if policy == 'candidate':
            candidate = Agent(seed, Config(**config))
        elif policy == 'untuned':
            candidate = Agent(seed, Config())
        else:
            candidate = BaselineAgent('balanced', seed)
        agents = opponents(kind, seed)
        agents.insert(seat, candidate)
        def observe(o, action, elapsed):
            if o['hero'] == seat:
                candidate_counts['decisions'] += 1
                candidate_counts['over_1ms'] += elapsed > 1000000
        blinds = None if schedule == 'doubling' else [(10,20),(15,30),(25,50),(40,80),(60,120),(100,200),(150,300),(250,500)]
        result = play_table(agents, seed, payouts=PAYOUTS[payout], blind_schedule=blinds,
                            observer=observe, decision_limit_ns=1000000 if enforce else None)
        candidate_counts['internal_fallbacks'] += getattr(candidate, 'errors', 0)
        rows.append(dict(seed=seed, seat=seat, points=result['points'][seat],
                         chips=result['stacks'][seat], won=int(result['stacks'][seat] == 16000),
                         survived=int(result['stacks'][seat] > 0), hands=result['hands'],
                         table_errors=result['errors'], table_invalid=result['invalid_actions'],
                         table_overruns=result['overruns']))
    return dict(scenario=scenario, policy=policy, rows=rows, candidate_counts=candidate_counts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default='artifacts/policy.json')
    p.add_argument('--tables', type=int, default=1000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--enforce-1ms', action='store_true')
    p.add_argument('--output', default='artifacts/heldout.json')
    args = p.parse_args()
    config = json.loads(Path(args.policy).read_text())['config']
    Agent(config=Config(**config))
    scenarios = []
    for kind in (*POPULATIONS.keys(), 'challenge', 'mirror'):
        for payout in ('winner', 'top3'):
            scenarios.append((kind, payout, 'doubling', len(scenarios)))
    for payout, schedule in (('top3', 'gentle'), ('flat3', 'doubling'), ('linear', 'doubling')):
        scenarios.append(('mixed', payout, schedule, len(scenarios)))
    jobs = []
    for scenario in scenarios:
        for policy in ('candidate', 'untuned', 'balanced'):
            for start in range(0, args.tables, 100):
                jobs.append((scenario, config, min(100, args.tables - start), start, policy, args.enforce_1ms))
    ctx = multiprocessing.get_context('fork' if 'fork' in multiprocessing.get_all_start_methods() else 'spawn')
    started = time.time()
    groups = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        for chunk in pool.map(task, jobs):
            key = (tuple(chunk['scenario']), chunk['policy'])
            group = groups.setdefault(key, dict(rows=[], candidate_counts={k: 0 for k in chunk['candidate_counts']}))
            group['rows'].extend(chunk['rows'])
            for k, v in chunk['candidate_counts'].items():
                group['candidate_counts'][k] += v
            if len(group['rows']) == args.tables:
                print(key, 'points', summary([r['points'] for r in group['rows']])['mean'], flush=True)
    results = []
    for scenario in scenarios:
        selected = {policy: groups[(scenario, policy)] for policy in ('candidate', 'untuned', 'balanced')}
        for group in selected.values():
            group['rows'].sort(key=lambda r: r['seed'])
            group['metrics'] = {k: summary([r[k] for r in group['rows']]) for k in ('points', 'chips', 'won', 'survived', 'hands')}
        candidate = selected['candidate']['rows']
        paired = {}
        for name in ('untuned', 'balanced'):
            ref = selected[name]['rows']
            assert [r['seed'] for r in ref] == [r['seed'] for r in candidate]
            paired[name] = {k: summary([a[k] - b[k] for a, b in zip(candidate, ref)]) for k in ('points', 'chips', 'won')}
        results.append(dict(scenario=scenario, policies=selected, paired_differences=paired))
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
              ('bot/agent.py','bot/equity.py','benchmarks/challenge.py','artifacts/equity_model.json.gz','artifacts/policy.json')}
    report = dict(split='heldout; not used for selection', tables_per_scenario=args.tables,
                  scenarios=len(scenarios), seconds=time.time() - started,
                  seating='Crosses all64candidate-seat/initial-button pairs per64tables; seed%8 button with seat=(table_index//8)%8.',
                  source_sha256=hashes, selected_config=config, timing_enforced=args.enforce_1ms,
                  limitations='Synthetic policy pool; shared equity errors in challenge/mirror; normal confidence intervals, no multiple-comparison correction. Paired decks diverge as elimination paths differ.',
                  results=results)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print('Saved', args.output, 'seconds', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
