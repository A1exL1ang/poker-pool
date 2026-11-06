"""Fit and export original, offline equity regression; no sklearn at runtime.

Run from workspace root. Training and held-out test datasets MUST use different
seeds. The test file is evaluated only after hyperparameters have been fixed.
"""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bot.features import VERSION, make_features


def dataset(path, feature_function=make_features, feature_count=58):
    import numpy as np
    rows = np.loadtxt(path, delimiter=',')
    x = np.empty((len(rows), feature_count), dtype=np.float32)
    y = rows[:, -1]
    for i, r in enumerate(rows):
        hole = [int(r[0]), int(r[1])]
        board = [int(c) for c in r[2:7] if c >= 0]
        x[i] = feature_function(hole, board, int(r[7]), float(r[8]), float(r[9]))
    return x, y, rows


def stats(y, pred):
    import numpy as np
    error = pred - y
    return {'n': len(y), 'mae': float(np.mean(np.abs(error))),
            'rmse': float(np.sqrt(np.mean(error ** 2))),
            'bias': float(np.mean(error)),
            'abs_error_p95': float(np.percentile(np.abs(error), 95)),
            'abs_error_p99': float(np.percentile(np.abs(error), 99)),
            'max_abs_error': float(np.max(np.abs(error)))}


def export(model, path, parameters, version=VERSION, feature_count=58):
    trees = []
    for stage in model._predictors:
        tree = []
        for n in stage[0].nodes:
            if n['is_leaf']:
                tree.append([-1, round(float(n['value']), 12), 0, 0])
            else:
                tree.append([int(n['feature_idx']), float(n['num_threshold']), int(n['left']), int(n['right'])])
        trees.append(tree)
    data = {'feature_version': version, 'feature_count': feature_count,
            'baseline': float(model._baseline_prediction[0, 0]), 'trees': trees,
            'parameters': parameters,
            'range_fractions': [0.1, 0.25, 0.5, 1.0],
            'pressure_levels': [0, 1, 2],
            'target': 'Monte Carlo expected pot share conditional on synthetic opponent ranges'}
    path.write_text(json.dumps(data, separators=(',', ':')))
    return data


def scalar_predict(data, row):
    value = data['baseline']
    for tree in data['trees']:
        node = tree[0]
        while node[0] >= 0:
            node = tree[node[2] if row[node[0]] <= node[1] else node[3]]
        value += node[1]
    return min(1., max(0., value))


def export_preflop(path, output):
    rows = list(csv.reader(open(path)))
    strength = [float(r[3]) for r in rows]
    combos = [6 if i < 13 else 4 if i < 91 else 12 for i in range(169)]
    percentiles = []
    for i, s in enumerate(strength):
        better = sum(combos[j] for j, t in enumerate(strength) if t > s)
        percentiles.append((better + 0.5 * combos[i]) / 1326)
    data = {'description': 'Original Monte Carlo preflop expected pot shares; ranges sorted by heads-up random equity',
            'range_fractions': [0.1, 0.25, 0.5, 1.0],
            'strength': strength, 'percentile': percentiles,
            'equity': [[float(x) for x in r[4:]] for r in rows]}
    output.write_text(json.dumps(data, separators=(',', ':')))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='training/train.csv')
    p.add_argument('--validation', default='training/validation.csv')
    p.add_argument('--extra-train', action='append', default=[])
    p.add_argument('--features-module', default='bot.features')
    p.add_argument('--test', default=None)
    p.add_argument('--preflop', default='training/preflop.csv')
    p.add_argument('--output', default='assets/equity_model.json')
    p.add_argument('--metrics', default='training/model_metrics.json')
    p.add_argument('--iterations', type=int, default=180)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--leaves', type=int, default=48)
    p.add_argument('--learning-rate', type=float, default=0.09)
    p.add_argument('--l2', type=float, default=2.0)
    args = p.parse_args()
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.perf_counter()
    fm = importlib.import_module(args.features_module)
    feature_function = fm.make_features
    feature_count = len(feature_function([0, 5], [10, 15, 20]))
    x, y, rows = dataset(args.train, feature_function, feature_count)
    for extra in args.extra_train:
        xe, ye, re = dataset(extra, feature_function, feature_count)
        x, y, rows = np.concatenate((x, xe)), np.concatenate((y, ye)), np.concatenate((rows, re))
    xv, yv, rv = dataset(args.validation, feature_function, feature_count)
    params = dict(max_iter=args.iterations, max_depth=args.depth,
                  max_leaf_nodes=args.leaves, learning_rate=args.learning_rate,
                  l2_regularization=args.l2, min_samples_leaf=30,
                  early_stopping=False, random_state=1616)
    model = HistGradientBoostingRegressor(**params).fit(x, y)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = export(model, output, params, fm.VERSION, feature_count)
    pv = np.clip(model.predict(xv), 0, 1)
    parity = max(abs(scalar_predict(data, row) - pred) for row, pred in zip(xv[:1000], pv[:1000]))
    if parity > 1e-8:
        raise RuntimeError(f'Export prediction mismatch {parity}')
    report = {'parameters': params, 'feature_version': fm.VERSION, 'feature_count': feature_count,
              'training_rows': len(y), 'validation': stats(yv, pv),
              'export_max_difference': parity, 'model_bytes': output.stat().st_size,
              'dataset_sha256': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for path in [args.train, args.validation] + args.extra_train},
              'validation_by_street': {}, 'validation_by_pressure': {}, 'validation_by_opponents': {}}
    for k in (3, 4, 5):
        pick = xv[:, 0] == k
        report['validation_by_street'][str(k)] = stats(yv[pick], pv[pick])
    for k in (0, 1, 2):
        pick = rv[:, 9] == k
        report['validation_by_pressure'][str(k)] = stats(yv[pick], pv[pick])
    for k in (1, 2, 3, 5, 7):
        pick = rv[:, 7] == k
        report['validation_by_opponents'][str(k)] = stats(yv[pick], pv[pick])
    if args.test:
        xt, yt, rt = dataset(args.test, feature_function, feature_count)
        pt = np.clip(model.predict(xt), 0, 1)
        report['held_out_test'] = stats(yt, pt)
        report['test_sha256'] = hashlib.sha256(Path(args.test).read_bytes()).hexdigest()
        report['test_calibration'] = []
        for lo in np.arange(0, 1, 0.1):
            pick = (pt >= lo) & (pt < lo + 0.1)
            if np.sum(pick):
                report['test_calibration'].append({'lower': float(lo), 'n': int(np.sum(pick)),
                    'mean_prediction': float(np.mean(pt[pick])), 'mean_target': float(np.mean(yt[pick]))})
    if Path(args.preflop).exists():
        export_preflop(args.preflop, output.parent / 'preflop.json')
    report['fit_export_seconds'] = time.perf_counter() - t0
    Path(args.metrics).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
