"""Reproducible LOCAL synthetic NLHE sit-and-go benchmark; no official claims.

Examples (from repository root)::
  python tools/benchmark.py --tables 100 --split tuning --lineup mixed --out measurements/tuning.json
  python tools/benchmark.py --tables 100 --split heldout --lineup mirror --out measurements/heldout.json
  python tools/benchmark.py --stress --tables 1000 --out measurements/stress.json

Tuning seeds are [0,10000); held-out seeds start at 100000. Keep held-out seeds
unused during policy selection. CI assumes independent seeded tables and uses a
normal approximation. ICM at hand cap supplies fractional finish distributions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import BASELINES, BASE_REFERENCE_PARAMS, make_baseline
from bot import Agent, DEFAULTS
from engine import tournament
from evaluator import evaluate

SYNTHETIC_PAYOUTS = {
    'wta': [1., 0., 0., 0., 0., 0., 0., 0.],
    'sng': [.5, .3, .2, 0., 0., 0., 0., 0.],
    'flat3': [1., 1., 1., 0., 0., 0., 0., 0.],
}
# Explicit synthetic schedule, NOT the unavailable official schedule.
SYNTHETIC_BLINDS = [(10, 20), (15, 30), (25, 50), (40, 80),
                    (60, 120), (100, 200), (150, 300), (250, 500)]


def quantile(values, q):
    if not values:
        return None
    index = (len(values)-1) * q
    low = int(index)
    high = min(len(values)-1, low+1)
    return values[low] + (values[high]-values[low]) * (index-low)


def summary(values):
    n = len(values)
    mean = statistics.fmean(values) if values else 0.
    se = statistics.stdev(values)/math.sqrt(n) if n > 1 else None
    return dict(n=n, mean=mean, stderr=se,
                ci95=[mean-1.96*se, mean+1.96*se] if se is not None else None)


def run_benchmark(args):
    if args.tables <= 0 or args.max_hands < 0:
        raise ValueError('Positive table count and nonnegative hand count required')
    start_seed = args.seed_start if args.seed_start is not None else (0 if args.split == 'tuning' else 100000)
    if args.split == 'tuning' and not 0 <= start_seed <= start_seed+args.tables-1 < 10000:
        raise ValueError('Tuning seeds must stay within [0,10000)')
    if args.split == 'heldout' and start_seed < 100000:
        raise ValueError('Held-out seeds must start >=100000')
    if args.lineup == 'mixed':
        lineup = ['station', 'nit', 'loose', 'maniac', 'random', 'value', 'value']
    elif args.lineup == 'mixed-mirror':
        lineup = ['station', 'nit', 'loose', 'maniac', 'random', 'value', 'mirror']
    else:
        lineup = args.lineup.split(',')
        if len(lineup) == 1:
            lineup *= 7
        if len(lineup) != 7 or any(name not in (*BASELINES, 'mirror') for name in lineup):
            raise ValueError('Lineup needs one style or exactly seven comma-separated styles')
    if 'mc' in lineup and args.enforce_time_limit:
        raise ValueError('MC is a deliberately slow benchmark opponent; disable timing enforcement and inspect focal overruns')
    if args.params:
        params = (json.loads(args.params) if args.params.lstrip().startswith('{') else
                  json.loads(Path(args.params).read_text(encoding='utf-8')))
    else:
        params = None
    blinds = json.loads(args.blinds) if args.blinds else SYNTHETIC_BLINDS
    payouts = json.loads(args.payouts_json) if args.payouts_json else SYNTHETIC_PAYOUTS[args.payouts]
    if len(payouts) != 8:
        raise ValueError('This benchmark uses eight seats')
    records, latencies, finish = [], [], [0.] * 8
    focal_actions = dict(fold=0, check=0, call=0, raise_=0)
    focal_errors = focal_timeouts = focal_internal = focal_decisions = 0
    all_errors = all_timeouts = all_internal = all_decisions = 0
    t0 = time.perf_counter()
    for offset in range(args.tables):
        seed = start_seed + offset
        focal = offset % 8  # Rotate focal seat rather than always assigning seat0.
        focal_agent = Agent(seed=seed*101+focal+9, params=params, model_path=args.model_path)
        agents, names = [], []
        villain = 0
        for seat in range(8):
            if seat == focal:
                agents.append(focal_agent)
                names.append('focal')
            else:
                name = lineup[(villain + offset//8) % 7]
                villain += 1
                agents.append(Agent(seed=seed*101+seat+9, model_path=args.model_path) if name == 'mirror'
                              else make_baseline(name, seed=seed*101+seat+9, model_path=args.model_path))
                names.append(name)
        result = tournament(agents, seed=seed, payouts=payouts, blind_schedule=blinds,
                            initial_stack=args.initial_stack, max_hands=args.max_hands,
                            evaluator=evaluate, enforce_time_limit=args.enforce_time_limit,
                            record_latencies=not args.no_timing)
        internal = [getattr(agent, 'errors', 0) for agent in agents]
        focal_errors += result['errors_by_seat'][focal]
        focal_timeouts += result['timeouts_by_seat'][focal]
        focal_internal += internal[focal]
        focal_decisions += result['decisions_by_seat'][focal]
        all_errors += result['errors']
        all_timeouts += result['timeouts']
        all_internal += sum(internal)
        all_decisions += result['decisions']
        latencies.extend(result['latencies_by_seat'][focal])
        for action in focal_actions:
            focal_actions[action] += result['actions_by_seat'][focal][action]
        for place in range(8):
            finish[place] += result['finish_probabilities'][focal][place]
        record = dict(seed=seed, focal_seat=focal, lineup=names, points=result['points'][focal],
                      chips=result['stacks'][focal], hands=result['hands'],
                      alive_at_cap=result['stacks'][focal] > 0 and result['hands'] == args.max_hands,
                      errors=result['errors'], timeouts=result['timeouts'], internal_fallbacks=sum(internal),
                      focal_errors=result['errors_by_seat'][focal], focal_internal_fallbacks=internal[focal],
                      focal_timeouts=result['timeouts_by_seat'][focal],
                      focal_decisions=result['decisions_by_seat'][focal])
        records.append(record)
        if args.progress and (offset+1) % args.progress == 0:
            print(f'{offset+1}/{args.tables} tables, mean={statistics.fmean(r["points"] for r in records):.6f}, '
                  f'errors={all_errors}+{all_internal} internal, timeouts={all_timeouts}', file=sys.stderr)
    latencies.sort()
    artifact_hashes = {}
    for name in ('bot.py', 'model.py', 'features.py', 'equity.py', 'evaluator.py', 'engine.py', 'baselines.py',
                 'assets/equity_model.json.gz', 'assets/equity_model.bin',
                 'assets/preflop.csv', 'assets/preflop_meta.json'):
        path = ROOT / name
        if path.is_file():
            artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.model_path and Path(args.model_path).is_file():
        artifact_hashes['selected_model:'+str(args.model_path)] = hashlib.sha256(Path(args.model_path).read_bytes()).hexdigest()
    effective_defaults = dict(DEFAULTS)
    effective_params = dict(effective_defaults)
    effective_params.update(params or {})
    latency = dict(n=len(latencies), p50_ns=quantile(latencies, .5), p95_ns=quantile(latencies, .95),
                   p99_ns=quantile(latencies, .99), p999_ns=quantile(latencies, .999),
                   max_ns=max(latencies) if latencies else None,
                   over_1ms=sum(t > 1_000_000 for t in latencies),
                   over_1ms_rate=sum(t > 1_000_000 for t in latencies)/len(latencies) if latencies else None)
    return dict(benchmark='independent-local-synthetic-NLHE-v1', official_compatibility=False,
                config=dict(tables=args.tables, split=args.split, seed_start=start_seed,
                            lineup=lineup, payouts=payouts, synthetic_payouts=not bool(args.payouts_json),
                            blind_schedule=blinds, synthetic_blinds=not bool(args.blinds),
                            max_hands=args.max_hands, initial_stack=args.initial_stack,
                            enforce_time_limit=args.enforce_time_limit,
                            benchmark_only_slow_opponents=['mc'] if 'mc' in lineup else [],
                            baseline_reference_params=dict(BASE_REFERENCE_PARAMS) if 'reference' in lineup else None,
                            params=params, effective_defaults=effective_defaults,
                            effective_params=effective_params,
                            model_path=args.model_path, stress=args.stress),
                runtime=dict(python=sys.version, implementation=platform.python_implementation(),
                             platform=platform.platform(), machine=platform.machine()),
                focal_points=summary([r['points'] for r in records]),
                focal_final_chips=summary([r['chips'] for r in records]),
                equal_share_reference=sum(payouts)/8,
                finish_distribution=[p/args.tables for p in finish],
                finish_distribution_semantics='observed elimination ranks; exact ICM probabilities for cap survivors',
                focal_diagnostics=dict(decisions=focal_decisions, engine_errors=focal_errors,
                                       internal_fallbacks=focal_internal, enforced_timeouts=focal_timeouts,
                                       actions=focal_actions),
                all_diagnostics=dict(decisions=all_decisions, engine_errors=all_errors,
                                     internal_fallbacks=all_internal, enforced_timeouts=all_timeouts),
                focal_latency=latency, wall_seconds=time.perf_counter()-t0,
                artifact_sha256=artifact_hashes, tables=records,
                limitations=['Synthetic heuristic opponent pool; no equilibrium or contest performance claim.',
                             'MC opponents deliberately exceed 1ms and are benchmark-only when present.',
                             'Normal CI may be inaccurate for small samples or rare large payouts.',
                             'OS scheduling and machine load affect wall-clock decision times.',
                             'Do not use held-out results to tune the policy and still call them held-out.'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tables', type=int, default=100)
    p.add_argument('--split', choices=['tuning', 'heldout'], default='heldout')
    p.add_argument('--seed-start', type=int)
    p.add_argument('--lineup', default='mixed', help='mixed, mixed-mirror, mirror, one baseline, or seven comma-separated styles')
    p.add_argument('--payouts', choices=SYNTHETIC_PAYOUTS, default='sng')
    p.add_argument('--payouts-json', help='Explicit eight-rank payout JSON, overrides synthetic preset')
    p.add_argument('--blinds', help='Explicit JSON [[sb,bb],...] per 25 hands; defaults to labeled synthetic schedule')
    p.add_argument('--params', help='Parameter JSON string or local JSON path')
    p.add_argument('--model-path')
    p.add_argument('--initial-stack', type=int, default=2000)
    p.add_argument('--max-hands', type=int, default=200)
    p.add_argument('--enforce-time-limit', action='store_true')
    p.add_argument('--no-timing', action='store_true')
    p.add_argument('--stress', action='store_true', help='Label a robustness run; use --tables 1000 for optional thousand-table stress')
    p.add_argument('--progress', type=int, default=25, help='Progress interval on stderr; zero disables')
    p.add_argument('--out', help='Write JSON here; otherwise emit on stdout')
    args = p.parse_args()
    result = run_benchmark(args)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text+'\n', encoding='utf-8')
        print(json.dumps({k: result[k] for k in ('focal_points', 'focal_diagnostics', 'all_diagnostics', 'focal_latency', 'wall_seconds')}, indent=2))
    else:
        print(text)
    if args.stress and (result['all_diagnostics']['engine_errors'] or result['all_diagnostics']['internal_fallbacks']):
        raise SystemExit('Stress run exposed errors; see diagnostics')


if __name__ == '__main__':
    main()
