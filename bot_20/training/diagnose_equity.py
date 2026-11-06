"""Inspect validation errors with only Python's standard library."""
import argparse
import ast
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokerbot.equity import EquityEstimator
from pokerbot.evaluator import card_text
from pokerbot.features import extract_features, FEATURE_NAMES


def rows(path):
    with open(path, "rb") as file:
        if file.read(8) != b"\x93NUMPY\x01\x00":
            raise ValueError("Expected generator's NPY v1 format")
        header = ast.literal_eval(file.read(struct.unpack("<H", file.read(2))[0]).decode().strip())
        if header["descr"] != "<f4" or header["shape"][1] != 13:
            raise ValueError("Unexpected label schema")
        return list(struct.iter_unpack("<13f", file.read()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/data/validation.npy")
    parser.add_argument("--model", default="pokerbot/assets/equity_model.json.gz")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--report", default="measurements/equity_validation_worst.json")
    args = parser.parse_args()
    model = EquityEstimator(args.model)
    if not model.model_loaded:
        raise RuntimeError("Trained model required")
    results = []
    for i, row in enumerate(rows(args.data)):
        hole = tuple(int(c) for c in row[:2])
        board = tuple(int(c) for c in row[2:2 + int(row[7])])
        features = extract_features(hole, board)
        prediction = model.predict(features, int(row[8]), row[9], row[10])
        results.append({"row": i, "hole": " ".join(map(card_text, hole)), "board": " ".join(map(card_text, board)),
                        "opponents": int(row[8]), "range_fraction": row[9], "pressure": row[10], "target": row[11],
                        "stderr": row[12], "prediction": prediction, "error": prediction - row[11],
                        "features": dict(zip(FEATURE_NAMES, features))})
    results.sort(key=lambda x: abs(x["error"]), reverse=True)
    report = {"purpose": "Validation diagnostics used for model development, not final evaluation", "model": args.model,
              "source": args.data, "rows": len(results), "worst": results[:args.top]}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": args.report, "top": [{k: v for k, v in r.items() if k != "features"} for r in results[:args.top]]}))


if __name__ == "__main__":
    main()
