"""Reproducible local tournament benchmark; synthetic schedules/payouts only.

Example: python benchmarks/tournament.py --tables 200 --seed 16000 --field mixed
Use disjoint --seed blocks for tuning and held-out evaluation. This runner never
reads other bots or sources: every opponent is independently authored locally.
"""
import argparse
import importlib
import hashlib
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines import Baseline, FIELDS, make_field
from simulator import play_table, SYNTHETIC_BLINDS

EXTRA_FIELDS = ('adversarial', 'calling', 'minraise', 'polar', 'selfplay')

# Labels and values are local experiment assumptions, not official point awards.
PAYOUTS = {
    'winner': [1, 0, 0, 0, 0, 0, 0, 0],
    'top3': [50, 30, 20, 0, 0, 0, 0, 0],
    'linear': [8, 7, 6, 5, 4, 3, 2, 1],
}


def fingerprints():
    paths = ['agent.py', 'bot/policy.py', 'bot/equity.py', 'bot/features.py', 'bot/features_base.py', 'bot/exact.py',
             'bot/evaluator.py', 'bot/icm.py', 'bot/_equity_compiled.py', 'simulator.py', 'baselines.py', 'baselines_stress.py',
             'assets/equity_model.json', 'assets/preflop.json', 'assets/preflop_narrow.json']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in paths if (ROOT / name).exists()}


def summarize(values):
    n = len(values)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else 0
    return dict(n=n, mean=mean, standard_error=se, ci95_normal=[mean - 1.96 * se, mean + 1.96 * se])


def run(tables=100, seed=16000, field='mixed', candidate='agent:Agent', payout='top3',
        max_hands=200, config=None, enforce_time_limit=False, observation_path=None, blind_schedule=None):
    if candidate.startswith('baseline:'):
        cls = lambda: Baseline(candidate.split(':', 1)[1])
    else:
        module, name = candidate.split(':', 1)
        constructor = getattr(importlib.import_module(module), name)
        cls = lambda: constructor(**(config or {}))
    source_hashes = fingerprints()
    candidate_points, candidate_chips = [], []
    hands, violations, over, decisions = [], [], 0, 0
    candidate_times = []
    action_counts = Counter()
    errors, fallbacks = 0, 0
    saved = []
    t0 = time.time()
    for table in range(tables):
        seat = table % 8
        if field == 'selfplay':
            agents = [cls() for _ in range(7)]
        elif field in EXTRA_FIELDS:
            from baselines_stress import make_stress_field
            agents = make_stress_field(field, seed + table)
        else:
            agents = make_field(field, seed + table)
        agents.insert(seat, cls())
        def observe(i, observation, action, duration):
            if i == seat:
                candidate_times.append(duration)
                action_counts[f"{observation['street']}:{action['action']}"] += 1
                if observation_path is not None and len(saved) < 10000:
                    saved.append(observation)
        result = play_table(agents, seed=seed + table, payouts=PAYOUTS[payout], max_hands=max_hands,
                            observe=observe, enforce_time_limit=enforce_time_limit, blind_schedule=blind_schedule)
        errors += getattr(agents[seat], 'errors', 0)
        fallbacks += getattr(agents[seat], 'fallbacks', 0)
        candidate_points.append(result.points[seat])
        candidate_chips.append(result.stacks[seat])
        hands.append(result.hands)
        violations.extend(v for v in result.violations if v['seat'] == seat)
        over += result.overruns_by_seat[seat]
        decisions += result.decisions_by_seat[seat]
    candidate_times.sort()
    def percentile(p):
        return candidate_times[min(len(candidate_times) - 1, int(p * len(candidate_times)))] / 1000 if candidate_times else 0
    report = dict(candidate=candidate, config=config or {}, field=field, tables=tables, seed_start=seed,
                  seed_end=seed + tables - 1, max_hands=max_hands, synthetic_payout_label=payout,
                  synthetic_payouts=PAYOUTS[payout], synthetic_blinds=blind_schedule or SYNTHETIC_BLINDS,
                  enforce_time_limit=enforce_time_limit, points=summarize(candidate_points),
                  equal_share_points=sum(PAYOUTS[payout]) / 8,
                  final_chips=summarize(candidate_chips), mean_hands=statistics.mean(hands),
                  candidate_violations=violations[:20], candidate_violation_count=len(violations),
                  candidate_decisions=decisions, candidate_overruns_1ms=over,
                  candidate_internal_errors=errors, candidate_internal_fallbacks=fallbacks,
                  candidate_action_counts=dict(action_counts),
                  file_sha256=source_hashes, files_changed_during_run=fingerprints() != source_hashes,
                  latency_us=dict(p50=percentile(.5), p95=percentile(.95), p99=percentile(.99), maximum=percentile(1)),
                  elapsed_seconds=time.time() - t0, per_table_points=candidate_points)
    if observation_path is not None:
        Path(observation_path).write_text(json.dumps(saved), encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables', type=int, default=100)
    p.add_argument('--seed', type=int, default=16000)
    p.add_argument('--field', choices=list(FIELDS) + list(EXTRA_FIELDS), default='mixed')
    p.add_argument('--candidate', default='agent:Agent')
    p.add_argument('--payout', choices=PAYOUTS, default='top3')
    p.add_argument('--max-hands', type=int, default=200)
    p.add_argument('--blinds', choices=('default', 'doubling'), default='default', help='Both later-blind schedules are synthetic assumptions')
    p.add_argument('--config', help='JSON object of candidate constructor kwargs')
    p.add_argument('--params', help='JSON object of policy parameters; shorthand for --config with params key')
    p.add_argument('--enforce-time-limit', action='store_true')
    p.add_argument('--observations', help='save up to 10000 hero decision observations as JSON')
    p.add_argument('--output', help='save JSON report')
    args = p.parse_args()
    config = json.loads(args.config) if args.config else {}
    if args.params:
        config['params'] = json.loads(args.params)
    report = run(tables=args.tables, seed=args.seed, field=args.field, candidate=args.candidate,
                 payout=args.payout, max_hands=args.max_hands,
                 config=config,
                 enforce_time_limit=args.enforce_time_limit, observation_path=args.observations,
                 blind_schedule=[(10 * 2 ** i, 20 * 2 ** i) for i in range(8)] if args.blinds == 'doubling' else None)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
