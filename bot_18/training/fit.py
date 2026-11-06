"""Fit/export compact stdlib-only equity models; no training dependencies at runtime.

Example: OMP_NUM_THREADS=8 python3 training/fit.py --train training/train.csv
  --heldout training/heldout.csv --preflop training/preflop.csv --iterations 180
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equity.features import FEATURE_COUNT, FEATURE_VERSION, features
from equity.preflop import class_index, RANGE_FRACTIONS


def load_data(path):
    import numpy as np
    raw = np.loadtxt(path, delimiter=",", skiprows=1)
    x = np.empty((len(raw), FEATURE_COUNT + 2), dtype=np.float32)
    started = time.monotonic()
    for i, row in enumerate(raw):
        hole = (int(row[0]), int(row[1]))
        board = tuple(int(c) for c in row[2:7] if c >= 0)
        x[i, :-2] = features(hole, board)
        x[i, -2:] = row[7:9]
        if i and i % 100000 == 0:
            print(f"encoded {i}/{len(raw)} in {time.monotonic()-started:.1f}s", flush=True)
    return x, raw[:, 9], raw


def export_model(model):
    trees = []
    for predictors in model._predictors:
        predictor = predictors[0]
        nodes = []
        for node in predictor.nodes:
            if node["is_leaf"]:
                nodes.append([-1, 0, 0, 0, float(node["value"])])
            else:
                nodes.append([int(node["feature_idx"]), float(node["num_threshold"]),
                              int(node["left"]), int(node["right"]), 0])
        trees.append(nodes)
    return {"format": 1, "feature_version": FEATURE_VERSION, "feature_count": FEATURE_COUNT + 2,
            "base": float(model._baseline_prediction[0, 0]), "trees": trees}


def predict_export(model, row):
    if "models" in model:
        model = model["models"][str(int(row[0]))]
    result = model["base"]
    for tree in model["trees"]:
        node = tree[0]
        while node[0] >= 0:
            node = tree[node[2] if row[node[0]] <= node[1] else node[3]]
        result += node[4]
    return min(1.0, max(0.0, result))


def metrics(y, p):
    import numpy as np
    error = p - y
    return {"n": len(y), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "p95_absolute": float(np.quantile(np.abs(error), .95)),
            "bias": float(np.mean(error))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--preflop")
    parser.add_argument("--premium")
    parser.add_argument("--output", default=str(ROOT / "equity" / "model.json"))
    parser.add_argument("--report", default=str(ROOT / "training" / "metrics.json"))
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--leaves", type=int, default=31)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--street-models", action="store_true")
    parser.add_argument("--compiled-output")
    parser.add_argument("--extra-train", action="append", default=[])
    parser.add_argument("--extra-weight", type=float, default=.5)
    args = parser.parse_args()
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    started = time.monotonic()
    x, y, raw_train = load_data(args.train)
    normal_rows = len(y)
    weights = np.ones(len(y))
    for extra_path in args.extra_train:
        extra_x,extra_y,_ = load_data(extra_path)
        x = np.concatenate((x,extra_x));y=np.concatenate((y,extra_y))
        weights=np.concatenate((weights,np.full(len(extra_y),args.extra_weight)))
    held_x, held_y, held_raw = load_data(args.heldout)
    print(f"encoded data {time.monotonic()-started:.1f}s; fitting", flush=True)
    monotonic = [0] * (FEATURE_COUNT + 1) + [-1]
    def new_model():
        return HistGradientBoostingRegressor(loss="squared_error", learning_rate=.075,
                                         max_iter=args.iterations, max_leaf_nodes=args.leaves,
                                         max_depth=args.depth, l2_regularization=.2,
                                         min_samples_leaf=35, early_stopping=False,
                                         monotonic_cst=monotonic, random_state=180181)
    if args.street_models:
        models={};prediction=np.zeros(len(held_y));train_prediction=np.zeros(min(10000,len(y)))
        export={"format":2,"feature_version":FEATURE_VERSION,"feature_count":FEATURE_COUNT+2,"models":{}}
        for street in (3,4,5):
            model=new_model();mask=x[:,0]==street;eval_mask=held_x[:,0]==street
            print(f"fitting street {street}: {mask.sum()} rows",flush=True)
            model.fit(x[mask],y[mask],sample_weight=weights[mask]);models[street]=model
            prediction[eval_mask]=np.clip(model.predict(held_x[eval_mask]),0,1)
            sample_mask=x[:len(train_prediction),0]==street
            train_prediction[sample_mask]=np.clip(model.predict(x[:len(train_prediction)][sample_mask]),0,1)
            export["models"][str(street)]=export_model(model)
    else:
        model=new_model();model.fit(x,y,sample_weight=weights)
        prediction=np.clip(model.predict(held_x),0,1)
        train_prediction=np.clip(model.predict(x[:10000]),0,1)
        export=export_model(model)
    reference = np.array([predict_export(export, row) for row in held_x[:512]])
    max_export_error = float(np.max(np.abs(reference - prediction[:512])))
    if max_export_error > 1e-10:
        raise AssertionError(f"export mismatch: {max_export_error}")
    report = {"method": "street-specific HistGradientBoostingRegressor" if args.street_models else "HistGradientBoostingRegressor", "train_rows": len(y),
              "normal_train_rows":normal_rows,"extra_train_files":[Path(p).name for p in args.extra_train],"extra_train_weight":args.extra_weight,
              "train_file": Path(args.train).name, "heldout_file": Path(args.heldout).name,
              "parameters": model.get_params(), "features": FEATURE_COUNT + 2,
              "heldout": metrics(held_y, prediction),
              "train_sample": metrics(y[:10000], train_prediction),
              "max_export_error": max_export_error, "by_street": {}, "by_range": {},
              "by_opponents": {}, "elapsed_seconds": time.monotonic()-started}
    for n in (3,4,5):
        mask = held_x[:,0] == n
        report["by_street"][str(n)] = metrics(held_y[mask], prediction[mask])
    for fraction in RANGE_FRACTIONS:
        mask = np.abs(held_raw[:,7]-fraction)<1e-7
        report["by_range"][str(fraction)] = metrics(held_y[mask], prediction[mask])
    for n in (1,2,3,5,7):
        mask = held_raw[:,8]==n
        report["by_opponents"][str(n)] = metrics(held_y[mask], prediction[mask])
    worst = np.argsort(np.abs(prediction-held_y))[-20:][::-1]
    report["largest_errors"] = [{"cards": held_raw[i,:7].astype(int).tolist(),
                                 "range": float(held_raw[i,7]), "opponents": int(held_raw[i,8]),
                                 "equity": float(held_y[i]), "prediction": float(prediction[i])}
                                for i in worst]
    Path(args.output).write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
    if args.compiled_output:
        from training.compile_model import compile_model
        compile_model(args.output,args.compiled_output)
        import runpy
        compiled_predict=runpy.run_path(args.compiled_output)["predict"]
        compiled_predictions=np.clip([compiled_predict(row) for row in held_x[:512]],0,1)
        report["max_compiled_error"]=float(np.max(np.abs(compiled_predictions-prediction[:512])))
        assert report["max_compiled_error"]<1e-10
    if args.preflop:
        raw = np.loadtxt(args.preflop, delimiter=",", skiprows=1)
        table = [[[None]*7 for _ in range(5)] for _ in range(169)]
        for row in raw:
            cls = class_index((int(row[0]),int(row[1])))
            ri = min(range(5), key=lambda j:abs(RANGE_FRACTIONS[j]-row[7]))
            table[cls][ri][int(row[8])-1] = float(row[9])
        assert all(v is not None for cls in table for rng in cls for v in rng)
        pf_path = Path(args.output).with_name("preflop.json")
        pf_path.write_text(json.dumps({"format":1,"fractions":RANGE_FRACTIONS,"table":table},separators=(",",":")),encoding="utf-8")
    if args.premium:
        export_premium(args.premium, Path(args.output).with_name("premium.json"))
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"heldout":report["heldout"],"by_street":report["by_street"],
                      "elapsed_seconds":report["elapsed_seconds"],"model_bytes":Path(args.output).stat().st_size}, indent=2))


def export_premium(input_path, output_path):
    import numpy as np
    raw = np.loadtxt(input_path, delimiter=",", skiprows=1)
    table = [[None,None] for _ in range(169)]
    for row in raw:
        cls = class_index((int(row[0]),int(row[1])))
        table[cls][int(-row[7])-1] = float(row[9])
    assert all(v is not None for pair in table for v in pair)
    Path(output_path).write_text(json.dumps({"format":1,"ranges":["QQ+/AK","KK+"],"table":table},separators=(",",":")),encoding="utf-8")


if __name__ == "__main__":
    main()
