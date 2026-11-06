"""Compare predeclared tree shapes on training/validation; never final test."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from pokerbot.features import FEATURE_NAMES, FEATURE_VERSION
from training.fit_equity import matrix, model_to_dict, pure_predictions, grouped_metrics, worst_cases, sha256, QUERY_FEATURES

CONFIGS = {
    "wide64": (False, 64, 255, 14, .18),
    "street48": (True, 48, 511, 16, .22),
    "street64": (True, 64, 255, 14, .18),
    "street96": (True, 96, 127, 12, .13),
    "street128": (True, 128, 127, 12, .11),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="training/data/train.npy")
    parser.add_argument("--validation", default="training/data/validation.npy")
    parser.add_argument("--preflop", default="pokerbot/assets/preflop.json")
    parser.add_argument("--candidates", default=",".join(CONFIGS))
    parser.add_argument("--output-dir", default="training/candidates")
    parser.add_argument("--report", default="measurements/equity_experiments.json")
    args = parser.parse_args()
    start = time.perf_counter()
    train, valid = np.load(args.train), np.load(args.validation)
    with open(args.preflop) as file:
        preflop = json.load(file)
    xt, xv = matrix(train, preflop), matrix(valid, preflop)
    metadata = {"train_sha256": sha256(args.train), "validation_sha256": sha256(args.validation), "train_rows": len(train),
                "validation_rows": len(valid), "feature_version": FEATURE_VERSION}
    results = []
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in args.candidates.split(","):
        split, iterations, leaves, depth, rate = CONFIGS[name]
        models = {}
        prediction = np.empty(len(valid))
        node_count = 0
        fit_start = time.perf_counter()
        max_difference = 0
        for street in ((3, 4, 5) if split else (0,)):
            train_mask = train[:, 7] == street if split else np.ones(len(train), dtype=bool)
            valid_mask = valid[:, 7] == street if split else np.ones(len(valid), dtype=bool)
            model = HistGradientBoostingRegressor(loss="squared_error", learning_rate=rate, max_iter=iterations, max_leaf_nodes=leaves,
                max_depth=depth, min_samples_leaf=30, l2_regularization=.2, early_stopping=False, random_state=671131)
            model.fit(xt[train_mask], train[train_mask, 11])
            exported = model_to_dict(model)
            models[str(street)] = exported
            p = model.predict(xv[valid_mask])
            prediction[valid_mask] = p
            sample = xv[valid_mask][:500]
            max_difference = max(max_difference, float(np.max(np.abs(pure_predictions(exported, sample) - p[:len(sample)]))))
            node_count += sum(map(len, exported["trees"]))
        if max_difference > 1e-10:
            raise AssertionError("Tree export parity failed")
        output = {"format_version": 2 if split else 1, "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
                  "query_features": QUERY_FEATURES, "training": {**metadata, "candidate": name, "configuration": CONFIGS[name]}}
        if split:
            output["models"] = models
        else:
            output.update(models["0"])
        path = destination / (name + ".json.gz")
        with open(path, "wb") as file:
            with gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed:
                compressed.write(json.dumps(output, separators=(",", ":"), allow_nan=False).encode())
        result = {"candidate": name, "configuration": CONFIGS[name], "model": str(path), "model_bytes": path.stat().st_size,
                  "nodes": node_count, "fit_seconds": time.perf_counter() - fit_start, "export_max_abs_difference": max_difference,
                  "validation": grouped_metrics(valid, np.clip(prediction, 0, 1))}
        result["worst_validation_cases"] = worst_cases(valid, np.clip(prediction, 0, 1))
        results.append(result)
        report = {"purpose": "Training/validation model selection only; final test untouched", "metadata": metadata,
                  "elapsed_seconds": time.perf_counter() - start, "candidates": results}
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"candidate": name, "validation": result["validation"]["all"], "fit_seconds": result["fit_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
