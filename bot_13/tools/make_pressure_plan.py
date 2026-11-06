"""A targeted TRAINING experiment for an exposed small-bet pressure weakness."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from lab.benchmark import LINEUPS, LOCAL_BLIND_SCENARIOS
from bot13.agent import DEFAULTS


def main():
    tasks = []
    for opening in (0.85, 1.0):
        for probe in (0.0, 0.35, 0.70):
            for wider in (False, True):
                for pending in (False, True):
                    name = f"o{round(opening * 100)}p{round(probe * 100)}r{int(wider)}a{int(pending)}"
                    cfg = {"open_scale": opening, "probe_frequency": probe,
                           "wide_threebet": wider, "assume_unacted_continue": pending}
                    for i, scenario in enumerate(("minpressure", "stress", "mixed")):
                        args = {"config": cfg, "reference_config": {}, "opponents": LINEUPS[scenario],
                                "tables": 192, "seed_start": 300000 + i * 10000,
                                "payouts": [50, 30, 20, 0, 0, 0, 0, 0],
                                "blind_schedule": LOCAL_BLIND_SCENARIOS["gentle"],
                                "enforce_deadline": True}
                        tasks.append({"name": name + "__" + scenario, "args": args})
    for name, cfg in (("base", {"probe_frequency": 0.0, "wide_threebet": False, "assume_unacted_continue": True}),
                      ("current", {})):
        tasks.append({"name": name + "__reference", "args": {
            "config": cfg, "reference_config": {**DEFAULTS, "probe_frequency": 0.0, "wide_threebet": False, "assume_unacted_continue": True},
            "opponents": ["reference"] * 7, "tables": 768, "seed_start": 330000,
            "payouts": [50, 30, 20, 0, 0, 0, 0, 0], "enforce_deadline": True}})
    result = {"phase": "targeted_training", "description": "No official payouts/schedule. Diagnose and refine pressure response.",
              "tasks": tasks, "reserved_heldout_seeds": ">=900000"}
    path = ROOT / "plans" / "pressure_v2.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(len(tasks), "tasks", sum(t["args"]["tables"] for t in tasks), "training tables")


if __name__ == "__main__":
    main()
