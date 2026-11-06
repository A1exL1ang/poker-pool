"""Frozen-candidate evaluation on independently seeded LOCAL sit-and-gos."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter

from simulator.benchmark import _one_table, capture_provenance, preflight_models, summarize
from simulator.scenarios import FIELDS, BLINDS, PAYOUTS


SCENARIOS = (
    ("robust", "medium", "top3"),
    ("robust", "fast", "winner"),
    ("robust", "slow", "flat3"),
    ("tight", "medium", "top3"),
    ("loose", "fast", "winner"),
    ("passive", "slow", "rank"),
    ("selfplay", "medium", "top3"),
    ("mixed", "slow", "top3"),
    ("mixed", "fast", "flat3"),
)


def task(item):
    scenario, index, options = item
    return scenario, _one_table((index, options))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="measurements/final_tournaments.json")
    parser.add_argument("--freeze", default="artifacts/final_manifest.json")
    parser.add_argument("--seed-start", type=int, default=190000000)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or args.tables < 8:
        parser.error("workers1..8 and at least8 tables per scenario required")
    frozen = json.loads(Path(args.freeze).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    for name, expected in frozen["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected["sha256"]:
            raise RuntimeError("Frozen file changed: " + name)
    groups = {s: [] for s in range(len(SCENARIOS))}
    tasks, options_list = [], []
    for s, (field, blinds, payout) in enumerate(SCENARIOS):
        options = {"candidate": "agent:Agent", "candidate_config": {},
                   "reference": "model_equity", "reference_config": {}, "opponent_config": {},
                   "opponents": ["agent:Agent"] * 7 if field == "selfplay" else FIELDS[field],
                   "max_hands": 200, "enforce_time": True, "allow_fallback_model": False,
                   "seating": "independent", "seed_start": args.seed_start + s * 100000,
                   "blinds": blinds, "payout": payout}
        preflight_models(options)
        options_list.append(options)
        tasks.extend((s, i, options) for i in range(args.tables))
    provenance = capture_provenance(options_list[0])
    started = perf_counter()
    if args.workers == 1:
        for scenario, row in map(task, tasks):
            groups[scenario].append(row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for scenario, row in pool.map(task, tasks, chunksize=4):
                groups[scenario].append(row)
    report = {"purpose": "Final independently seeded local evaluation after source/asset freeze",
              "limitations": "Hypothetical payouts/later blinds, own baselines and own bot copies; not an unknown-contest-field forecast. Wall-time timeout fallback enforced.",
              "frozen_manifest": frozen, "provenance": provenance, "tables_per_scenario": args.tables,
              "elapsed_seconds": perf_counter() - started, "scenarios": []}
    for s, (field, blinds, payout) in enumerate(SCENARIOS):
        report["scenarios"].append({"field": field, "blind_name": blinds, "payout_name": payout,
                                    "blind_levels": BLINDS[blinds], "payouts": PAYOUTS[payout],
                                    "options": options_list[s], "summary": summarize(groups[s], "independent"),
                                    "rows": groups[s]})
    # A second check ensures no file changed during a supposedly frozen run.
    for name, expected in frozen["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected["sha256"]:
            raise RuntimeError("Frozen file changed during evaluation: " + name)
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "tables": args.tables * len(SCENARIOS),
                      "elapsed_seconds": report["elapsed_seconds"],
                      "scenarios": [{k: v for k, v in s.items() if k not in ("rows", "options", "blind_levels", "payouts")} for s in report["scenarios"]]}, indent=2))
    return report


if __name__ == "__main__":
    main()
