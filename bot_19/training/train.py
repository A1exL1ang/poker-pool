"""Train on synthetic labels and select iteration count using validation only.

Training dependencies: numpy, xgboost. Exported inference uses only stdlib.
The XGBoost native train/get_dump API is documented at
https://xgboost.readthedocs.io/en/stable/python/python_api.html
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import struct
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poker_eval import FEATURE_NAMES
from equity_features import MODEL_FEATURE_NAMES, extend_features


def read_data(path):
    data = np.fromfile(path, dtype="<f4").reshape(-1, len(FEATURE_NAMES) + 1)
    if not np.all(np.isfinite(data)):
        raise ValueError("nonfinite training data")
    if not np.all((data[:, -1] >= 0) & (data[:, -1] <= 1)):
        raise ValueError("invalid equity labels")
    return data


def metrics(y, pred):
    error = np.asarray(pred) - y
    return {"rows": int(len(y)), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))), "bias": float(np.mean(error)),
            "p95_absolute_error": float(np.quantile(np.abs(error), .95)),
            "p99_absolute_error": float(np.quantile(np.abs(error), .99)),
            "max_absolute_error": float(np.max(np.abs(error)))}


def flat_tree(tree):
    nodes = {}
    def walk(node):
        node_id = node["nodeid"]
        if "leaf" in node:
            nodes[node_id] = [-1, node["leaf"]]
        else:
            feature = int(node["split"][1:])
            # XGBoost compares float32 values. Reconstruct its exact threshold;
            # decimal dump rounding must not change equality at anchor values.
            threshold = struct.unpack("f", struct.pack("f", node["split_condition"]))[0]
            nodes[node_id] = [feature, threshold, node["yes"], node["no"]]
            for child in node["children"]:
                walk(child)
    walk(tree)
    return [nodes[i] for i in range(max(nodes) + 1)]


def predict_export(model, row):
    if model.get("derived") and len(row) == len(FEATURE_NAMES):
        row = extend_features(row)
    row = list(row)
    row[41], row[42] = struct.unpack("ff", struct.pack("ff", row[41], row[42]))
    result = model["base"]
    for tree in model["trees"]:
        n = 0
        while tree[n][0] >= 0:
            e = tree[n]
            n = e[2] if row[e[0]] < e[1] else e[3]
        result += tree[n][1]
    return min(.999, max(.001, result))


def file_info(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--output", default="assets/equity_model.json")
    parser.add_argument("--report", default="measurements/equity_training.json")
    parser.add_argument("--trees", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=.085)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=191911)
    args = parser.parse_args()
    threads = min(8, max(1, args.threads))
    train = read_data(args.train)
    valid = read_data(args.valid)
    exported = {"schema": 1, "feature_names": list(MODEL_FEATURE_NAMES),
                "input_feature_names": list(FEATURE_NAMES), "derived_version": 1, "streets": {},
                "range_model": "smooth_preflop_percentile_prior_times_continuation_power_3aggression"}
    report = {"training": file_info(args.train), "validation": file_info(args.valid),
              "selection": "per-street validation RMSE; test labels never read",
              "parameters": vars(args), "python": platform.python_version(), "xgboost": xgb.__version__,
              "numpy": np.__version__, "streets": {}}
    started = time.perf_counter()
    for street in (3, 4, 5):
        t = train[train[:, 0] == street]
        v = valid[valid[:, 0] == street]
        xt = np.asarray([extend_features(row) for row in t[:, :-1]], dtype=np.float32)
        xv = np.asarray([extend_features(row) for row in v[:, :-1]], dtype=np.float32)
        dt = xgb.DMatrix(xt, label=t[:, -1], nthread=threads)
        dv = xgb.DMatrix(xv, label=v[:, -1], nthread=threads)
        params = {"objective": "reg:squarederror", "tree_method": "hist", "max_depth": args.depth,
                  "eta": args.learning_rate, "min_child_weight": 30, "lambda": 3,
                  "subsample": .9, "colsample_bytree": 1.0, "max_bin": 128,
                  "seed": args.seed + street, "nthread": threads, "base_score": .5}
        booster = xgb.train(params, dt, num_boost_round=args.trees,
                            evals=[(dv, "validation")], early_stopping_rounds=24, verbose_eval=32)
        rounds = booster.best_iteration + 1
        pred = np.clip(booster.predict(dv, iteration_range=(0, rounds)), .001, .999)
        raw = booster.get_dump(dump_format="json")[:rounds]
        model = {"base": .5, "derived": True, "trees": [flat_tree(json.loads(tree)) for tree in raw]}
        exported["streets"][str(street)] = model
        # Check all exported branches on a fixed validation subset.
        python_pred = [predict_export(model, row) for row in v[:300, :-1]]
        parity = float(np.max(np.abs(np.array(python_pred) - pred[:300])))
        if parity > 2e-5:
            raise RuntimeError(f"export parity failure: {parity}")
        result = {"train_rows": len(t), "trees": rounds, "metrics": metrics(v[:, -1], pred),
                  "export_max_difference": parity, "strata": {}}
        for label, mask in (("range_width_le_020", v[:, 41] <= .20),
                            ("range_width_ge_060", v[:, 41] >= .60),
                            ("aggression_ge_070", v[:, 42] >= .70),
                            ("no_aggression", v[:, 42] == 0),
                            ("pair_or_highcard", v[:, 6] <= 1),
                            ("strong_made", v[:, 6] >= 4)):
            if np.any(mask):
                result["strata"][label] = metrics(v[mask, -1], pred[mask])
        report["streets"][str(street)] = result
        print(json.dumps({"street": street, **result}, indent=2), flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(exported, separators=(",", ":")) + "\n")
    report["model"] = file_info(args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
