"""Lightweight review of saved held-out summaries; performs no gameplay work."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result_dir, audit = a.run / "results", a.run / "audit"
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((result_dir / "plan.json").read_text(encoding="utf-8"))
    assert manifest["phase"] == "final_heldout"
    assert not manifest["errors"] and len(manifest["completed"]) == len(plan["tasks"])
    settings = {t["name"]: t["args"] for t in plan["tasks"]}
    output = {"created_utc": datetime.now(timezone.utc).isoformat(), "phase": "final_heldout",
              "selected_variant": plan["selected_variant"], "heldout_used_for_tuning": False,
              "all_blinds_and_payout_scenarios_local_not_official": True,
              "total_tables": sum(t["args"]["tables"] for t in plan["tasks"]),
              "release_tables": 0, "release_decisions": 0, "release_deadline_fallbacks": 0,
              "release_internal_fallbacks": 0, "release_native_errors": 0,
              "scenarios": {}, "diagnostic_baseline_and_fallback": {}}
    for task in manifest["completed"]:
        name, scenario = task["name"].split("__", 1)
        summary = task["summary"]
        args = settings[task["name"]]
        payouts = args.get("payouts")
        metric = summary["candidate_points"] if payouts else summary["candidate_chip_share"]
        denominator = sum(payouts) if payouts else 1.0
        row = {"tables": metric["count"], "payouts": payouts,
               "mean_score_share": metric["mean"] / denominator,
               "score_share_ci95": [v / denominator for v in metric["ci95_normal"]],
               "points": summary["candidate_points"], "chips": summary["candidate_chips"],
               "latency": summary["candidate_latency"], "internal": summary["candidate_internal"]}
        if name == "release":
            output["scenarios"][scenario] = row
            output["release_tables"] += row["tables"]
            output["release_decisions"] += row["latency"]["count"]
            output["release_deadline_fallbacks"] += row["latency"]["fallbacks"].get("deadline", 0)
            output["release_internal_fallbacks"] += row["internal"].get("fallbacks", 0)
            output["release_native_errors"] += row["internal"].get("native_errors", 0)
            assert not row["latency"]["errors"]
        else:
            output["diagnostic_baseline_and_fallback"][task["name"]] = row
    for name, filename in (("tests", "test_summary.json"), ("package", "package_validation.json"),
                           ("serial_replay", "serial_replay.json"), ("stress_timing", "stress_timing.json")):
        output[name] = json.loads((audit / filename).read_text(encoding="utf-8"))
    # One measured release task carries source hashes. Verify the collected
    # runtime files against that frozen experiment; documentation can evolve.
    measured = json.loads((result_dir / "release__mixed.json").read_text(encoding="utf-8"))
    hashes = measured["metadata"]["source_sha256"]
    local_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in hashes}
    output["frozen_runtime_source_matches_workspace"] = local_hashes == hashes
    output["runtime_source_sha256"] = hashes
    assert output["frozen_runtime_source_matches_workspace"]
    output["release_deadline_fallback_rate"] = output["release_deadline_fallbacks"] / output["release_decisions"]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Heldout tables:", output["total_tables"], "release tables:", output["release_tables"])
    print("Release decisions:", output["release_decisions"], "deadline fallbacks:", output["release_deadline_fallbacks"],
          "internal fallbacks:", output["release_internal_fallbacks"], "native errors:", output["release_native_errors"])
    for name, row in output["scenarios"].items():
        print(name, "score share", round(row["mean_score_share"], 5), "CI95", row["score_share_ci95"])
    print("serial replay:", output["serial_replay"]["summary"])
    print("tests:", output["tests"])


if __name__ == "__main__":
    main()
