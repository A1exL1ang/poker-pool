"""Selection scores normalized by each explicitly supplied local payout pool."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.directory / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((a.directory / "manifest.json").read_text(encoding="utf-8"))
    args = {task["name"]: task["args"] for task in plan["tasks"]}
    rows = {}
    for task in manifest["completed"]:
        name, scenario = task["name"].split("__", 1)
        cfg = args[task["name"]]
        points = cfg.get("payouts")
        s = task["summary"]
        denominator = sum(points) if points is not None else 1.0
        metric = s["candidate_points"] if points is not None else s["candidate_chip_share"]
        rows.setdefault(name, {"config": cfg["config"], "scenarios": {}})["scenarios"][scenario] = {
            "score_share": metric["mean"] / denominator,
            "standard_error": metric["standard_error"] / denominator,
            "chips": s["candidate_chips"], "points": s["candidate_points"],
            "latency": s["candidate_latency"], "tables": metric["count"]}
    result = []
    for name, value in rows.items():
        scores = value["scenarios"]
        overall = statistics.fmean(v["score_share"] for v in scores.values())
        strong = statistics.fmean(scores[k]["score_share"] for k in ("minpressure", "reference", "pressure_reference"))
        result.append({"variant": name, "mean_score_share": overall, "strong_pool_mean": strong,
                       "config": value["config"], "scenarios": scores})
    result.sort(key=lambda r: -(0.5 * r["mean_score_share"] + 0.5 * r["strong_pool_mean"]))
    output = {"phase": plan["phase"], "selection_metric": "equal weight mean of all local scenarios and mean of three strong-reference scenarios",
              "errors": manifest["errors"], "ranking": result}
    a.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for row in result:
        print(row["variant"], "all", round(row["mean_score_share"], 5), "strong", round(row["strong_pool_mean"], 5),
              {k: round(v["score_share"], 4) for k, v in row["scenarios"].items()})


if __name__ == "__main__":
    main()
