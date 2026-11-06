"""Bounded, reproducible DEVELOPMENT tuning; never use held-out seeds here.

Usage: python benchmarks/tune.py --tables 96 --seed 160000 --output benchmarks/tuning.json
Candidate selection on these tables makes every table tuning data. Run a separate
fresh seed block for held-out validation after freezing strategy and assets.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.tournament import run, summarize, fingerprints

VARIANTS = [
    ('default', {}),
    ('tighter_open', {'open_scale': .85}),
    ('wider_open', {'open_scale': 1.15}),
    ('tighter_call', {'call_margin': .05}),
    ('looser_call', {'call_margin': 0.0}),
    ('less_pressure', {'pressure_scale': .7}),
    ('more_pressure', {'pressure_scale': 1.3}),
    ('thin_value', {'value_threshold': .53}),
    ('strong_value', {'value_threshold': .62}),
    ('fewer_bluffs', {'bluff_scale': .6}),
    ('more_bluffs', {'bluff_scale': 1.4}),
    ('small_bets', {'bet_fraction': .50}),
    ('large_bets', {'bet_fraction': .80}),
    ('more_raises', {'raise_threshold': .70}),
    ('fewer_raises', {'raise_threshold': .80}),
]


def tune(tables=96, seed=160000, fields=('mixed', 'tight', 'aggressive'),
         variants=None, output='benchmarks/tuning.json', payout='top3'):
    variants = VARIANTS if variants is None else variants
    if not variants or variants[0][1]:
        raise ValueError('First variant must be default with empty params')
    hashes = fingerprints()
    reports = []
    control = {}
    for name, params in variants:
        field_reports = []
        aggregate_differences = []
        for i, field in enumerate(fields):
            report = run(tables=tables, seed=seed + i * 10000, field=field,
                         config={'params': params, 'strict': True}, payout=payout)
            if report['files_changed_during_run'] or fingerprints() != hashes:
                raise RuntimeError('Source/assets changed during experiment; results are not a frozen-policy comparison')
            if name == variants[0][0]:
                control[field] = report['per_table_points']
            differences = [a - b for a, b in zip(report['per_table_points'], control[field])]
            aggregate_differences.extend(differences)
            report['paired_difference_from_default'] = summarize(differences)
            field_reports.append(report)
            print(name, field, 'points', round(report['points']['mean'], 3),
                  'paired difference', round(statistics.mean(differences), 3), flush=True)
        reports.append(dict(name=name, params=params, fields=field_reports,
                            equal_weight_mean=statistics.mean(r['points']['mean'] for r in field_reports),
                            pooled_paired_difference=summarize(aggregate_differences)))
        artifact = dict(purpose='development tuning; not held-out evidence',
                        file_sha256=hashes, tables_per_field=tables, seed=seed,
                        fields=list(fields), payout=payout, variants=reports,
                        note='Normal CIs are descriptive; no correction for multiple candidate selection. Synthetic opponents/schedules/payouts are not contest guarantees.')
        Path(output).write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    return artifact


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables', type=int, default=96)
    p.add_argument('--seed', type=int, default=160000)
    p.add_argument('--fields', default='mixed,tight,aggressive')
    p.add_argument('--variants', help='JSON file of [[name,params], ...], default first')
    p.add_argument('--output', default='benchmarks/tuning.json')
    p.add_argument('--payout', choices=['winner', 'top3', 'linear'], default='top3')
    args = p.parse_args()
    variants = json.loads(Path(args.variants).read_text()) if args.variants else None
    tune(tables=args.tables, seed=args.seed, fields=args.fields.split(','),
         variants=variants, output=args.output, payout=args.payout)


if __name__ == '__main__':
    main()
