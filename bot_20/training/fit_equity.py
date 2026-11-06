"""Fit/export the offline equity model; this file is never imported in play.

Requires numpy and scikit-learn on the training machine. Training and validation
files must use distinct generator seeds. The final evaluation file is consumed
only by ``evaluate_equity.py`` after all selection is frozen.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from pokerbot.equity import hand_class
from pokerbot.evaluator import card_text
from pokerbot.features import extract_features, FEATURE_NAMES, FEATURE_VERSION

QUERY_FEATURES = ["opponents", "range_fraction", "pressure", "preflop_percentile"]


def sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def matrix(raw, preflop):
    x = np.empty((len(raw), len(FEATURE_NAMES) + 4), dtype=np.float32)
    percentiles = preflop["percentiles"]
    for i, row in enumerate(raw):
        hole = (int(row[0]), int(row[1]))
        board = tuple(int(c) for c in row[2:2 + int(row[7])])
        x[i, :-4] = extract_features(hole, board)
        x[i, -4:] = (row[8], row[9], row[10], percentiles[hand_class(hole)])
    return x


def metrics(y, prediction, stderr=None):
    error = prediction - y
    result = {"n": int(len(y)), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
              "bias": float(error.mean()), "p95_abs_error": float(np.quantile(np.abs(error), .95)), "max_abs_error": float(np.max(np.abs(error)))}
    if stderr is not None:
        noise = float(np.mean(stderr ** 2))
        result["monte_carlo_noise_rmse"] = noise ** .5
        result["noise_corrected_rmse_estimate"] = max(0.0, result["rmse"] ** 2 - noise) ** .5
    return result


def grouped_metrics(raw, prediction):
    result = {"all": metrics(raw[:, 11], prediction, raw[:, 12]), "by_street": {}, "by_opponents": {}, "by_pressure": {}}
    for n in (3, 4, 5):
        mask = raw[:, 7] == n
        if mask.any():
            result["by_street"][str(n)] = metrics(raw[mask, 11], prediction[mask], raw[mask, 12])
    for n in range(1, 8):
        mask = raw[:, 8] == n
        if mask.any():
            result["by_opponents"][str(n)] = metrics(raw[mask, 11], prediction[mask], raw[mask, 12])
    for low, high in ((0, .01), (.01, .5), (.5, .999), (.999, 1.001)):
        mask = (raw[:, 10] >= low) & (raw[:, 10] < high)
        if mask.any():
            result["by_pressure"][f"{low:g}-{high:g}"] = metrics(raw[mask, 11], prediction[mask], raw[mask, 12])
    return result


def worst_cases(raw, prediction, count=20):
    indices = np.argsort(np.abs(prediction - raw[:, 11]))[-count:][::-1]
    return [{"row": int(i), "hole": " ".join(card_text(int(c)) for c in raw[i, :2]),
             "board": " ".join(card_text(int(c)) for c in raw[i, 2:2 + int(raw[i, 7])]),
             "opponents": int(raw[i, 8]), "range_fraction": float(raw[i, 9]), "pressure": float(raw[i, 10]),
             "target": float(raw[i, 11]), "stderr": float(raw[i, 12]), "prediction": float(prediction[i]),
             "error": float(prediction[i] - raw[i, 11])} for i in indices]


def model_to_dict(model):
    trees = []
    for iteration in model._predictors:
        if len(iteration) != 1:
            raise ValueError("Expected one regression tree per iteration")
        nodes = []
        for node in iteration[0].nodes:
            if int(node["is_categorical"]):
                raise ValueError("Categorical tree export is unsupported")
            if int(node["is_leaf"]):
                nodes.append([-1, 0, 0, 0, float(node["value"])])
            else:
                nodes.append([int(node["feature_idx"]), float(node["num_threshold"]), int(node["left"]), int(node["right"]), 0])
        trees.append(nodes)
    return {"base": float(model._baseline_prediction[0, 0]), "trees": trees}


def export_model(model, path, metadata):
    result = {"format_version": 1, "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES), "query_features": QUERY_FEATURES,
              **model_to_dict(model), "training": metadata}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, separators=(",", ":"), allow_nan=False).encode("utf-8")
    # A fixed gzip timestamp makes rebuild comparisons independent of wall time.
    with open(path, "wb") as file:
        with gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed:
            compressed.write(encoded)
    return result


def pure_predictions(export, x):
    output = []
    for row in x:
        value = export["base"]
        for tree in export["trees"]:
            node = tree[0]
            while node[0] >= 0:
                node = tree[node[2] if row[node[0]] <= node[1] else node[3]]
            value += node[4]
        output.append(value)
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--preflop", required=True)
    parser.add_argument("--output", default="pokerbot/assets/equity_model.json.gz")
    parser.add_argument("--report", default="measurements/equity_validation.json")
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=.08)
    parser.add_argument("--min-leaf", type=int, default=30)
    parser.add_argument("--seed", type=int, default=671131)
    args = parser.parse_args()
    if Path(args.train).resolve() == Path(args.validation).resolve():
        raise ValueError("Training and validation data must differ")
    start = time.perf_counter()
    with open(args.preflop, encoding="utf-8") as file:
        preflop = json.load(file)
    training = np.load(args.train, allow_pickle=False)
    validation = np.load(args.validation, allow_pickle=False)
    xtrain, xvalid = matrix(training, preflop), matrix(validation, preflop)
    model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=args.learning_rate, max_iter=args.iterations,
        max_leaf_nodes=args.leaves, max_depth=args.depth, min_samples_leaf=args.min_leaf, l2_regularization=.2,
        early_stopping=False, random_state=args.seed)
    fit_start = time.perf_counter()
    model.fit(xtrain, training[:, 11])
    fit_seconds = time.perf_counter() - fit_start
    metadata = {"train_sha256": sha256(args.train), "validation_sha256": sha256(args.validation), "preflop_sha256": sha256(args.preflop),
                "train_rows": int(len(training)), "validation_rows": int(len(validation)), "parameters": vars(args),
                "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "fit_seconds": fit_seconds}
    export = export_model(model, args.output, metadata)
    raw_prediction = model.predict(xvalid)
    reference = pure_predictions(export, xvalid[:min(2000, len(xvalid))])
    max_export_error = float(np.max(np.abs(reference - raw_prediction[:len(reference)])))
    if max_export_error > 1e-10:
        raise AssertionError(f"Export differs from sklearn by {max_export_error}")
    prediction = np.clip(raw_prediction, 0, 1)
    report = {"purpose": "Model-selection validation; not final held-out evaluation", "metadata": metadata,
              "validation": grouped_metrics(validation, prediction), "export_max_abs_difference": max_export_error,
              "worst_validation_cases": worst_cases(validation, prediction),
              "trees": len(export["trees"]), "tree_nodes": sum(map(len, export["trees"])), "model_bytes": Path(args.output).stat().st_size,
              "elapsed_seconds": time.perf_counter() - start}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": args.report, "validation": report["validation"]["all"], "model_bytes": report["model_bytes"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
