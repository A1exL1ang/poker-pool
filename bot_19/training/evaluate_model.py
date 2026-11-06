"""Locked held-out evaluation; never fits or changes model parameters."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train import file_info, metrics, predict_export, read_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--model", default="assets/equity_model.json")
    parser.add_argument("--report", default="measurements/equity_heldout.json")
    args = parser.parse_args()
    data = read_data(args.test)
    model = json.loads(Path(args.model).read_text())
    report = {"model": file_info(args.model), "heldout": file_info(args.test),
              "meaning": "equity approximation error under declared synthetic range model; not strategic strength", "streets": {}}
    t0 = time.perf_counter()
    for street in (3, 4, 5):
        rows = data[data[:, 0] == street]
        pred = np.array([predict_export(model["streets"][str(street)], row) for row in rows[:, :-1]])
        report["streets"][str(street)] = metrics(rows[:, -1], pred)
    report["evaluation_seconds"] = time.perf_counter() - t0
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
