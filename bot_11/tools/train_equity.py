"""Fit/export equity models. Training only: requires numpy and scikit-learn.

Example: python -m tools.train_equity --train train.bin --test holdout.bin
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from pokerbot.features import features, FEATURE_NAMES
from pokerbot.equity import LABELS


def load(path):
    data = np.fromfile(path, dtype=np.dtype([('n', 'u1'), ('cards', 'u1', 7), ('y', '<f4', 10)]))
    x = np.asarray([features(tuple(int(c) for c in row['cards'][:2]), tuple(int(c) for c in row['cards'][2:2 + int(row['n'])])) for row in data], dtype=np.float32)
    return x, data['y'].copy()


def export(model, path):
    chunks = [struct.pack('<8sII', b'BOT11EQ1', len(model.estimators_), 10)]
    for estimator in model.estimators_:
        tree = estimator.tree_
        leaf = tree.children_left < 0
        leafids = np.cumsum(leaf) - 1
        fs = np.where(leaf, -1, tree.feature).astype('i1')
        left = np.where(leaf, leafids, tree.children_left).astype('<i4')
        right = tree.children_right.astype('<i4')
        vals = np.clip(np.rint(tree.value[leaf, :, 0] * 65535), 0, 65535).astype('<u2')
        chunks.extend([struct.pack('<II', tree.node_count, int(leaf.sum())), fs.tobytes(), tree.threshold.astype('<f4').tobytes(), left.tobytes(), right.tobytes(), vals.tobytes()])
    raw = b''.join(chunks)
    path.write_bytes(zlib.compress(raw, 6))
    return {'raw_bytes': len(raw), 'compressed_bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--out', default='assets/equity.bin.z')
    p.add_argument('--trees', type=int, default=64)
    p.add_argument('--leaf', type=int, default=10)
    p.add_argument('--depth', type=int, default=19)
    p.add_argument('--jobs', type=int, default=8)
    args = p.parse_args()
    started = time.monotonic()
    x, y = load(args.train)
    xt, yt = load(args.test)
    print('loaded', x.shape, xt.shape, flush=True)
    model = ExtraTreesRegressor(n_estimators=args.trees, min_samples_leaf=args.leaf, max_depth=args.depth, max_features=1.0, n_jobs=args.jobs, random_state=110639)
    model.fit(x, y)
    yp = model.predict(xt)
    report = {'train_states': len(x), 'heldout_states': len(xt), 'features': list(FEATURE_NAMES), 'labels': list(LABELS), 'parameters': model.get_params(),
              'mae': dict(zip(LABELS, np.mean(abs(yp-yt), axis=0).tolist())),
              'rmse': dict(zip(LABELS, np.sqrt(np.mean((yp-yt)**2, axis=0)).tolist())),
              'by_street': {}, 'source_train_sha256': hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
              'source_holdout_sha256': hashlib.sha256(Path(args.test).read_bytes()).hexdigest()}
    for n in (3, 4, 5):
        mask = xt[:, 0] == n
        report['by_street'][str(n)] = dict(zip(LABELS, np.mean(abs(yp[mask]-yt[mask]), axis=0).tolist()))
    report.update(export(model, Path(args.out)))
    report['seconds'] = time.monotonic() - started
    Path('benchmarks/equity_model.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed('benchmarks/equity_validation.npz', x=xt, y=yt, predicted=yp)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
