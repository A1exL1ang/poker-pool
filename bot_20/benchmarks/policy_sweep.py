"""Predeclared, small policy-parameter experiment on development seeds only.

Tables are paired by deals and opponents; each candidate is run independently.
Uniformly weighted scenarios are local sensitivity assumptions, not contest
scoring. The runner never touches the final-evaluation seed range.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
from time import perf_counter

from simulator.benchmark import _one_table, summarize
from simulator.scenarios import FIELDS


CONFIGS = {
    "base": {},
    "tight_open": {"open_scale": .85},
    "wide_open": {"open_scale": 1.15},
    "cautious": {"call_margin": .05, "pressure_scale": 1.15, "value_margin": .10, "bluff_scale": .6},
    "aggressive": {"open_scale": 1.1, "call_margin": .01, "value_margin": .055, "bluff_scale": 1.3, "pressure_scale": .85},
    "less_bluff": {"bluff_scale": .4},
    "thin_value": {"value_margin": .04},
    "no_adaptation": {"adaptation": False},
}

SCENARIOS = (
    ("robust", "medium", "top3"),
    ("tight", "fast", "flat3"),
    ("mixed", "fast", "winner"),
    ("loose", "slow", "top3"),
)


def work(task):
    name, scenario, index, options = task
    return name, scenario, _one_table((index, options))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, default=128, help="per candidate per scenario")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--names", default=",".join(CONFIGS))
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--output", default="measurements/policy_sweep.json")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8 or args.tables < 8 or args.seed_start >= 8200000:
        parser.error("bounded workers1..8, tables>=8 and development seeds required")
    from simulator.benchmark import capture_provenance
    provenance = capture_provenance({"candidate": "agent:Agent", "opponents": sorted({p for f in FIELDS.values() for p in f})})
    tasks = []
    names = args.names.split(",")
    for name in names:
        config = CONFIGS[name]
        for s, (field, blinds, payout) in enumerate(SCENARIOS):
            options = {"candidate": "agent:Agent", "candidate_config": config,
                       "reference": None, "reference_config": {}, "opponent_config": {},
                       "opponents": FIELDS[field], "max_hands": 200,
                       "enforce_time": False, "allow_fallback_model": False,
                       "seating": "independent", "seed_start": args.seed_start + s * 100000,
                       "blinds": blinds, "payout": payout}
            tasks.extend((name, s, i, options) for i in range(args.tables))
    started = perf_counter()
    groups = {name: {s: [] for s in range(len(SCENARIOS))} for name in names}
    if args.workers == 1:
        results = map(work, tasks)
        for name, s, row in results:
            groups[name][s].append(row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for name, s, row in pool.map(work, tasks, chunksize=4):
                groups[name][s].append(row)
    report = {"purpose": "Development-only selection; no final seeds used", "configs": {k: CONFIGS[k] for k in names},
              "scenarios": SCENARIOS, "tables_per_scenario": args.tables, "seed_start": args.seed_start,
              "provenance": provenance, "elapsed_seconds": perf_counter() - started,
              "candidates": {}}
    for name in names:
        summaries = [summarize(groups[name][s], "independent") for s in range(len(SCENARIOS))]
        candidate = {"scenarios": summaries, "pooled_mean": statistics.fmean(x["score"]["mean"] for x in summaries)}
        if "base" in groups:
            deltas = []
            by_scenario = []
            for s in range(len(SCENARIOS)):
                base = {r["index"]: r["score"] for r in groups["base"][s]}
                d = [r["score"] - base[r["index"]] for r in groups[name][s]]
                deltas.extend(d)
                by_scenario.append(statistics.fmean(d))
            candidate["paired_mean_delta_vs_base"] = statistics.fmean(deltas)
            candidate["paired_standard_error"] = statistics.stdev(deltas) / len(deltas) ** .5
            candidate["scenario_deltas_vs_base"] = by_scenario
        report["candidates"][name] = candidate
    report["rows"] = groups
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "elapsed_seconds": report["elapsed_seconds"],
                      "candidates": report["candidates"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
