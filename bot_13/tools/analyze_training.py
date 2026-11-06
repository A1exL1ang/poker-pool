"""Summarize declared training metrics; never reinterpret local payouts as official."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    manifest = json.loads((a.directory / "manifest.json").read_text(encoding="utf-8"))
    rows = {}
    for task in manifest["completed"]:
        variant, scenario = task["name"].split("__", 1)
        s = task["summary"]
        points = s["candidate_points"]
        score = points["mean"] / 100 if points is not None else s["candidate_chip_share"]["mean"]
        rows.setdefault(variant, {})[scenario] = {"score_share": score,
            "chips": s["candidate_chips"]["mean"],
            "points": points["mean"] if points else None,
            "latency": s["candidate_latency"]}
    ranking = sorted(({"variant": name, "mean_score_share": statistics.fmean(r["score_share"] for r in values.values()),
                       "scenarios": values} for name, values in rows.items()), key=lambda x: -x["mean_score_share"])
    for row in ranking:
        print(row["variant"], round(row["mean_score_share"], 5),
              {k: round(v["score_share"], 4) for k, v in row["scenarios"].items()})
    result = {"phase": manifest["phase"], "selection_metric": "mean across five declared local scenarios; high selection uncertainty",
              "errors": manifest["errors"], "ranking": ranking}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
