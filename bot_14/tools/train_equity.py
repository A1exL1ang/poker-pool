"""Train/export an original compact equity surrogate. Training dependencies only.

Gameplay uses only Python's standard library. The external validation file is
never used in fitting or early stopping; fixed hyperparameters and the training
split determine the model. A later untouched test set audits final selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time


def prepare_ranks(path, output):
    src = json.loads(Path(path).read_text())
    counts = src["combination_counts"]
    if "uniform_equity" in src:
        values = [x[0] for x in src["uniform_equity"]]
    else:
        values = [sum(p * n for p, n in zip(row, counts)) / sum(counts) for row in src["equity"]]
    ranks = [0.] * 169
    before = 0
    for i in sorted(range(169), key=values.__getitem__, reverse=True):
        ranks[i] = (before + counts[i] / 2) / sum(counts)
        before += counts[i]
    Path(output).write_text("\n".join(str(x) for x in ranks) + "\n")


def read_data(path):
    import numpy as np
    with open(path, "rb") as f:
        if f.read(8) != b"B14EQD1\0":
            raise ValueError("unknown dataset format")
        rows, nf, trials, seed = struct.unpack("<4I", f.read(16))
        data = np.fromfile(f, dtype="<f4").reshape(rows, nf + 1)
    return data[:, :-1], data[:, -1], dict(rows=rows, trials=trials, seed=seed, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def export(model):
    trees = []
    for iteration in model._predictors:
        ns = iteration[0].nodes
        trees.append([[-1, float(n["value"])] if n["is_leaf"] else [int(n["feature_idx"]), float(n["num_threshold"]), int(n["left"]), int(n["right"])] for n in ns])
    return dict(bias=float(model._baseline_prediction[0, 0]), trees=trees)


def compact_predict(model, x):
    out = model["bias"]
    for tree in model["trees"]:
        node = tree[0]
        while node[0] >= 0:
            node = tree[node[2] if x[node[0]] <= node[1] else node[3]]
        out += node[1]
    return min(1., max(0., out))


def metrics(y, pred):
    import numpy as np
    errors = pred - y
    return dict(rows=len(y), mae=float(np.mean(np.abs(errors))), rmse=float(np.mean(errors ** 2) ** .5), bias=float(np.mean(errors)), p95_absolute_error=float(np.quantile(np.abs(errors), .95)), p99_absolute_error=float(np.quantile(np.abs(errors), .99)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-ranks")
    p.add_argument("--ranks-output")
    p.add_argument("--train")
    p.add_argument("--validation")
    p.add_argument("--evaluate", help="audit an already exported model without retraining")
    p.add_argument("--output", default="assets/equity_model.json")
    p.add_argument("--report", default="measurements/equity_validation.json")
    p.add_argument("--iterations", type=int, default=220)
    p.add_argument("--leaves", type=int, default=31)
    p.add_argument("--derived", action="store_true")
    args = p.parse_args()
    if args.prepare_ranks:
        prepare_ranks(args.prepare_ranks, args.ranks_output)
        return
    import numpy as np
    start = time.perf_counter()
    xv, yv, vm = read_data(args.validation)
    models = {}
    report = dict(validation=vm, range_model="top-q preflop; top-m current strength with draw promotion; residual weak/bluff weight=.10", labels="conditional showdown pot share; exact HU river, Monte Carlo other cases")
    if args.evaluate:
        artifact = json.loads(Path(args.evaluate).read_text())
        models = artifact["models"]
        args.derived = artifact["feature_count"] > 47
    def expanded(x):
        if not args.derived:
            return x
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pokerbot.derived import augment
        return np.array([augment(tuple(row[:44]), *row[44:47]) for row in x], dtype=np.float32)
    if args.derived:
        xv = expanded(xv)
    if not args.evaluate:
        from sklearn.ensemble import HistGradientBoostingRegressor
        import sklearn
        xt, yt, tm = read_data(args.train)
        xt = expanded(xt)
        report.update(training=tm, sklearn=sklearn.__version__)
        for street in (3, 4, 5):
            mask = xt[:, 0] == street
            model = HistGradientBoostingRegressor(max_iter=args.iterations, learning_rate=.065, max_leaf_nodes=args.leaves, max_depth=12, min_samples_leaf=30, l2_regularization=.03, early_stopping=True, validation_fraction=.1, n_iter_no_change=25, random_state=140014 + street)
            model.fit(xt[mask], yt[mask])
            compact = export(model)
            models[str(street)] = compact
            # An exact parity check protects against version-specific export
            # internals and against accidental omission of learning-rate scales.
            selected = xv[xv[:, 0] == street][:250]
            expected = np.clip(model.predict(selected), 0, 1)
            actual = np.array([compact_predict(compact, x) for x in selected])
            error = float(np.max(np.abs(expected - actual)))
            if error > 1e-10:
                raise RuntimeError(f"export parity failed: {error}")
            print(json.dumps(dict(street=street, trees=len(compact["trees"]), export_max_error=error)), flush=True)
        artifact = dict(format="bot_14_equity_v1", feature_count=xt.shape[1], features="pokerbot.equity.FEATURE_NAMES + preflop_fraction,postflop_fraction,opponents; optional pokerbot.derived.augment", models=models, training=tm, training_parameters=dict(iterations=args.iterations, leaves=args.leaves, learning_rate=.065, min_samples_leaf=30, l2=.03), range_model=report["range_model"])
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(artifact, separators=(",", ":")))
    pred = np.array([compact_predict(models[str(int(x[0]))], x) for x in xv])
    report["overall"] = metrics(yv, pred)
    report["by_street"] = {str(s): metrics(yv[xv[:, 0] == s], pred[xv[:, 0] == s]) for s in (3, 4, 5)}
    report["by_opponents"] = {str(n): metrics(yv[xv[:, 46] == n], pred[xv[:, 46] == n]) for n in (1, 2, 3, 4, 7)}
    report["by_category"] = {str(n): metrics(yv[xv[:, 1] == n], pred[xv[:, 1] == n]) for n in range(9) if np.any(xv[:, 1] == n)}
    report["seconds"] = time.perf_counter() - start
    report["model_sha256"] = hashlib.sha256(Path(args.evaluate or args.output).read_bytes()).hexdigest()
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
