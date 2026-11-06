"""Evaluate an already-frozen exported model, without fitting or sklearn.

Example: python training/evaluate.py --test training/test.csv \
    --output benchmarks/equity_test.json --latency-repeats 2
"""
import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bot.equity import postflop, MODEL_AVAILABLE, PREFLOP_AVAILABLE
from bot.evaluator import evaluate


def quantile(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int((len(s) - 1) * q))]


def summary(rows):
    err = [r['prediction'] - r['target'] for r in rows]
    return {'n': len(err), 'mae': statistics.mean(abs(x) for x in err),
            'rmse': math.sqrt(statistics.mean(x * x for x in err)),
            'bias': statistics.mean(err),
            'abs_error_p95': quantile([abs(x) for x in err], .95),
            'abs_error_p99': quantile([abs(x) for x in err], .99),
            'max_abs_error': max(abs(x) for x in err)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--test', required=True)
    p.add_argument('--split', choices=('validation', 'held_out'), default='held_out')
    p.add_argument('--output', default='benchmarks/equity_test.json')
    p.add_argument('--latency-repeats', type=int, default=1)
    args = p.parse_args()
    if not MODEL_AVAILABLE or not PREFLOP_AVAILABLE:
        raise RuntimeError('Evaluation requires both packaged assets')
    source = Path(args.test)
    rows = []
    for r in csv.reader(source.open()):
        hole = [int(r[0]), int(r[1])]
        board = [int(x) for x in r[2:7] if int(x) >= 0]
        rows.append({'hole': hole, 'board': board, 'opponents': int(r[7]),
                     'range_fraction': float(r[8]), 'pressure': float(r[9]),
                     'target': float(r[10]), 'category': evaluate(hole + board) >> 20})
    elapsed = []
    for repeat in range(args.latency_repeats):
        for r in rows:
            t = time.perf_counter_ns()
            prediction = postflop(r['hole'], r['board'], r['range_fraction'], r['opponents'], r['pressure'])
            elapsed.append(time.perf_counter_ns() - t)
            r['prediction'] = prediction
    report = {'description': args.split + ' frozen model evaluation; synthetic ranges and Monte Carlo labels',
              'environment': {'python': sys.version, 'platform': platform.platform()},
              'test_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256((ROOT / 'assets/equity_model.json').read_bytes()).hexdigest(),
              'overall': summary(rows), 'by_street': {}, 'by_pressure': {},
              'by_opponents': {}, 'by_category': {}, 'calibration': []}
    for key, values, field in [('by_street', (3, 4, 5), 'board'),
                              ('by_pressure', (0, 1, 2), 'pressure'),
                              ('by_opponents', (1, 2, 3, 5, 7), 'opponents'),
                              ('by_category', range(9), 'category')]:
        for v in values:
            subset = [r for r in rows if (len(r[field]) if field == 'board' else r[field]) == v]
            if subset:
                report[key][str(v)] = summary(subset)
    for i in range(10):
        subset = [r for r in rows if i / 10 <= r['prediction'] < (i + 1) / 10 or (i == 9 and r['prediction'] == 1)]
        if subset:
            report['calibration'].append({'lower': i / 10, 'n': len(subset),
                'mean_prediction': statistics.mean(r['prediction'] for r in subset),
                'mean_target': statistics.mean(r['target'] for r in subset)})
    report['latency_microseconds'] = {'n': len(elapsed), 'first': elapsed[0] / 1000,
        'median': statistics.median(elapsed) / 1000, 'p95': quantile(elapsed, .95) / 1000,
        'p99': quantile(elapsed, .99) / 1000, 'p999': quantile(elapsed, .999) / 1000,
        'max': max(elapsed) / 1000, 'over_1ms': sum(t > 1_000_000 for t in elapsed)}
    report['worst_20'] = sorted(rows, key=lambda r: abs(r['prediction'] - r['target']), reverse=True)[:20]
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('worst_20', 'calibration')}, indent=2))


if __name__ == '__main__':
    main()
