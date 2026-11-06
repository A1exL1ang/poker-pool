"""Offline training only: numpy/sklearn required, never imported by bot.

python tools/train_equity.py --data data/train_*.csv --out assets/equity_model.json.gz
Data split uses independent generator seeds/shards; final holdout separate file.
"""
import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features import equity_features, N_FEATURES


def load(paths):
    xs, ys = [], []
    for path in paths:
        a = np.loadtxt(path, delimiter=',', skiprows=1)
        for row in a:
            hole = tuple(int(v) for v in row[:2])
            board = tuple(int(v) for v in row[2:7] if v >= 0)
            xs.append(equity_features(hole, board))
        ys.extend(a[:, 7:])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', nargs='+', required=True)
    p.add_argument('--holdout', nargs='+', required=True)
    p.add_argument('--out', default='assets/equity_model.json.gz')
    p.add_argument('--report', default='measurements/equity_model.json')
    p.add_argument('--trees', type=int, default=40)
    p.add_argument('--depth', type=int, default=16)
    p.add_argument('--leaf', type=int, default=12)
    a = p.parse_args()
    started = time.time()
    x, y = load(a.data)
    tx, ty = load(a.holdout)
    model = ExtraTreesRegressor(n_estimators=a.trees, max_depth=a.depth,
        min_samples_leaf=a.leaf, max_features=1.0, n_jobs=8, random_state=91847)
    model.fit(x, y)
    pred = model.predict(tx)
    with open(a.data[0], newline='') as f:
        targets = next(csv.reader(f))[7:]
    metadata = {'train_rows': len(y), 'holdout_rows': len(ty), 'features': N_FEATURES,
                'train_files': a.data, 'holdout_files': a.holdout,
                'trees': a.trees, 'max_depth': a.depth, 'min_leaf': a.leaf,
                'targets': targets,
                'mae': np.abs(pred-ty).mean(axis=0).tolist(),
                'rmse': np.sqrt(((pred-ty)**2).mean(axis=0)).tolist(),
                'elapsed_seconds': time.time()-started}
    metadata['by_street'] = {str(n): {'rows': int((tx[:, 0]==n).sum()),
        'mae': np.abs(pred[tx[:, 0]==n]-ty[tx[:, 0]==n]).mean(axis=0).tolist()}
        for n in (3, 4, 5)}
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append([[int(t.feature[i]), round(float(t.threshold[i]), 7),
                       int(t.children_left[i]), int(t.children_right[i])]
                      if t.feature[i] >= 0 else
                      [-1, [round(float(v), 7) for v in t.value[i, :, 0]]]
                      for i in range(t.node_count)])
    Path(a.out).parent.mkdir(exist_ok=True, parents=True)
    with gzip.open(a.out, 'wt', encoding='utf-8', compresslevel=9) as f:
        json.dump({'metadata': metadata, 'trees': trees}, f, separators=(',', ':'))
    Path(a.report).parent.mkdir(exist_ok=True, parents=True)
    Path(a.report).write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
