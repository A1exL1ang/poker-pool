"""Freeze one selected policy and create final held-out tasks; never tune on them."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot13.agent import DEFAULTS
from lab.benchmark import LINEUPS, LOCAL_BLIND_SCENARIOS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection-report", type=Path, required=True)
    p.add_argument("--variant", required=True)
    a = p.parse_args()
    report = json.loads(a.selection_report.read_text(encoding="utf-8"))
    chosen = next(row for row in report["ranking"] if row["variant"] == a.variant)
    selected = chosen["config"]
    selection = json.loads((ROOT / "plans" / "selection_v3.json").read_text(encoding="utf-8"))
    base = selection["base_reference_config"]
    pressure = selection["pressure_reference_config"]
    unseen = {**DEFAULTS, "open_scale": 1.10, "call_margin": 0.03, "value_threshold": 0.58,
              "bluff_frequency": 0.18, "bet_scale": 0.58, "range_threshold_scale": 0.90,
              "range_bluff": 0.12, "samples": 128, "wide_threebet": True,
              "assume_unacted_continue": False, "probe_frequency": 0.50, "pushfold": True}
    payouts = [50, 30, 20, 0, 0, 0, 0, 0]
    scenarios = [
        ("mixed", LINEUPS["mixed"], "gentle", payouts, base),
        ("minpressure", LINEUPS["minpressure"], "gentle", payouts, base),
        ("stress", LINEUPS["stress"], "gentle", payouts, base),
        ("reference", ["reference"] * 7, "gentle", payouts, base),
        ("pressure_reference", ["reference"] * 7, "gentle", payouts, pressure),
        ("equity", LINEUPS["equity"], "doubling", payouts, base),
        ("ladder", LINEUPS["mixed"], "gentle", [7, 6, 5, 4, 3, 2, 1, 0], base),
        ("tight", LINEUPS["tight"], "gentle", None, base),
        ("unseen_reference", ["reference"] * 7, "doubling", payouts, unseen),
    ]
    tasks = []
    for i, (scenario, opponents, blind, points, ref) in enumerate(scenarios):
        args = {"config": selected, "reference_config": ref,
                "opponents": opponents, "tables": 1024, "seed_start": 910000 + i * 10000,
                "payouts": points, "blind_schedule": LOCAL_BLIND_SCENARIOS[blind],
                "enforce_deadline": True}
        tasks.append({"name": "release__" + scenario, "args": args})
        if scenario in ("mixed", "minpressure", "reference", "pressure_reference"):
            tasks.append({"name": "baseline__" + scenario, "args": {**args, "config": base, "tables": 512}})
    for i, scenario in enumerate(("mixed", "minpressure")):
        tasks.append({"name": "python_fallback__" + scenario, "args": {
            "config": {**selected, "native": False}, "reference_config": base,
            "opponents": LINEUPS[scenario], "tables": 128, "seed_start": 1010000 + i * 10000,
            "payouts": payouts, "enforce_deadline": True}})
    policy = {"variant": a.variant, "config": selected,
              "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "selection_source": str(a.selection_report),
              "criterion": report["selection_metric"],
              "heldout_used_for_selection": False}
    (ROOT / "assets" / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    plan = {"phase": "final_heldout", "selected_variant": a.variant,
            "policy_frozen_before_heldout": True, "tasks": tasks,
            "all_scenarios_local_not_official": True}
    (ROOT / "plans" / "heldout_v4.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print("Froze", a.variant, ";", len(tasks), "tasks;", sum(t["args"]["tables"] for t in tasks), "held-out tables")


if __name__ == "__main__":
    main()
