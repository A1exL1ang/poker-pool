"""Separate selection/validation seeds; final held-out seeds remain >=900000."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.benchmark import LINEUPS, LOCAL_BLIND_SCENARIOS
from bot13.agent import DEFAULTS


def main():
    base = {**DEFAULTS, "probe_frequency": 0.0, "wide_threebet": False,
            "assume_unacted_continue": True, "pushfold": False}
    stable = {**base, "wide_threebet": True}
    pressure = {**base, "probe_frequency": 0.70, "wide_threebet": True,
                "assume_unacted_continue": False}
    variants = {
        "stable": stable,
        "pressure": pressure,
        "broad": {**pressure, "range_threshold_scale": 0.75, "range_bluff": 0.16},
        "small": {**stable, "bet_scale": 0.48},
        "bluffs": {**pressure, "bluff_frequency": 0.25},
        "accurate": {**pressure, "samples": 192, "fewway_samples": 96, "multiway_samples": 32},
    }
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
    ]
    tasks = []
    for name, config in variants.items():
        for endgame in (False, True):
            variant = name + ("_pf" if endgame else "")
            for i, (scenario, opponents, blind, points, ref_cfg) in enumerate(scenarios):
                args = {"config": {**config, "pushfold": endgame},
                        "reference_config": ref_cfg, "opponents": opponents,
                        "tables": 256, "seed_start": 610000 + i * 10000,
                        "payouts": points, "blind_schedule": LOCAL_BLIND_SCENARIOS[blind],
                        "enforce_deadline": True}
                tasks.append({"name": variant + "__" + scenario, "args": args})
    result = {"phase": "selection_validation", "description": "Declared local scenarios; final policy chosen using this validation set.",
              "selection_metric": "0.5*mean(all eight normalized scenarios) + 0.5*mean(minpressure, reference, pressure_reference); chosen before inspecting validation results",
              "tasks": tasks, "reserved_heldout_seeds": ">=900000", "base_reference_config": base,
              "pressure_reference_config": pressure}
    path = ROOT / "plans" / "selection_v3.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(len(tasks), "tasks", sum(t["args"]["tables"] for t in tasks), "selection tables")


if __name__ == "__main__":
    main()
