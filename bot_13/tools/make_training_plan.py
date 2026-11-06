"""Explicit local training plan. Final held-out seeds begin at 900000."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.benchmark import LINEUPS, LOCAL_BLIND_SCENARIOS

VARIANTS = {
    "base": {},
    "open_wide": {"open_scale": 1.15},
    "open_tight": {"open_scale": 0.85},
    "call_loose": {"call_margin": -0.01},
    "call_tight": {"call_margin": 0.05},
    "value_lower": {"value_threshold": 0.56},
    "value_higher": {"value_threshold": 0.67},
    "bet_large": {"bet_scale": 0.90},
    "bet_small": {"bet_scale": 0.48},
    "bluff_more": {"bluff_frequency": 0.24},
    "bluff_less": {"bluff_frequency": 0.02},
    "ranges_broad": {"range_threshold_scale": 0.75, "range_bluff": 0.16},
    "ranges_tight": {"range_threshold_scale": 1.15, "range_bluff": 0.04},
    "no_adapt": {"adapt": False},
    "no_icm": {"risk_scale": 0.0},
    "extra_icm": {"risk_scale": 1.5},
    "balanced": {"open_scale": 1.05, "call_margin": 0.03,
                 "value_threshold": 0.59, "bet_scale": 0.75,
                 "bluff_frequency": 0.09, "range_bluff": 0.10},
    "conservative": {"open_scale": 0.9, "call_margin": 0.04,
                     "value_threshold": 0.65, "bet_scale": 0.65},
}


def main():
    scenarios = [
        ("mixed", LINEUPS["mixed"], "gentle", None),
        ("equity", LINEUPS["equity"], "gentle", [50, 30, 20, 0, 0, 0, 0, 0]),
        ("aggressive", LINEUPS["aggressive"], "doubling", [50, 30, 20, 0, 0, 0, 0, 0]),
        ("reference", ["reference"] * 7, "gentle", [50, 30, 20, 0, 0, 0, 0, 0]),
        ("station", LINEUPS["station"], "gentle", None),
    ]
    tasks = []
    for variant, cfg in VARIANTS.items():
        for i, (name, opponents, blind, payouts) in enumerate(scenarios):
            tasks.append({"name": variant + "__" + name, "args": {
                "config": cfg, "reference_config": {},
                "tables": 64, "seed_start": 200000 + i * 10000,
                "opponents": opponents, "blind_schedule": LOCAL_BLIND_SCENARIOS[blind],
                "payouts": payouts, "enforce_deadline": True,
            }})
    result = {"phase": "training", "description": "Explicitly LOCAL scenarios; never official payouts or blind levels.",
              "selection_metric": "mean across scenarios of chip share, or supplied points/sum(payouts)",
              "reserved_heldout_seeds": ">=900000; never used for tuning", "tasks": tasks}
    (ROOT / "plans").mkdir(exist_ok=True)
    (ROOT / "plans" / "training_v1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(len(tasks), "tasks,", sum(t["args"]["tables"] for t in tasks), "training tables")


if __name__ == "__main__":
    main()
