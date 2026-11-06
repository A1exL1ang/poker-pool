"""Replay diverse LOCAL legal observations; measure wall time including tails.

JSON decoding and Agent construction are outside the timed act() contract.
No garbage-collector suppression, process priority changes, or timing censoring.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter_ns, process_time_ns

from agent import Agent
from pokerbot.interface import is_legal, validate_observation
from simulator.baselines import make_baseline
from simulator.benchmark import load_policy
from simulator.engine import play_tournament
from simulator.scenarios import BLINDS, FIELDS, PAYOUTS


def percentiles(values):
    values = sorted(values)
    n = len(values)
    return {"count": n, "mean_us": statistics.fmean(values) / 1000 if n else 0,
            "p50_us": values[min(n - 1, int(.5 * n))] / 1000 if n else 0,
            "p95_us": values[min(n - 1, int(.95 * n))] / 1000 if n else 0,
            "p99_us": values[min(n - 1, int(.99 * n))] / 1000 if n else 0,
            "p999_us": values[min(n - 1, int(.999 * n))] / 1000 if n else 0,
            "max_us": max(values, default=0) / 1000,
            "over_1ms": sum(x > 1_000_000 for x in values)}


def collect(tables, limit):
    observations = []
    per_seat = [0] * 8
    for t in range(tables):
        field = list(FIELDS)[t % len(FIELDS)]
        names = FIELDS[field] + ["random" if t % 3 == 0 else "model_equity"]
        agents = [load_policy(name, 8200000 + t * 13 + i) for i, name in enumerate(names)]
        def observe(obs):
            # All baseline seats are candidates for diverse full-information sets.
            # Avoid preferentially collecting only early streets after a limit.
            if len(observations) < limit:
                validate_observation(obs)
                observations.append(obs)
                per_seat[obs["hero"]] += 1
        play_tournament(agents, seed=8200000 + t, max_hands=200,
                        blind_levels=BLINDS[("slow", "medium", "fast")[t % 3]],
                        payouts=PAYOUTS[("top3", "winner", "flat3", "rank")[t % 4]],
                        on_observation=observe)
        if len(observations) >= limit:
            break
    return observations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, default=16)
    parser.add_argument("--limit", type=int, default=12000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--allow-fallback-model", action="store_true")
    parser.add_argument("--output", default="measurements/latency.json")
    args = parser.parse_args(argv)
    started = perf_counter_ns()
    bot = Agent(seed=811223)
    initialization = perf_counter_ns() - started
    flags = {"preflop_loaded": bot.estimator.preflop_loaded,
             "model_loaded": bot.estimator.model_loaded}
    if not args.allow_fallback_model and not all(flags.values()):
        raise RuntimeError("Trained assets are missing; benchmark refused to imply final performance")
    corpus = collect(args.tables, args.limit)
    wall, cpu = [], []
    streets = {s: [] for s in range(4)}
    illegal = 0
    overruns = []
    for repeat in range(args.repeats):
        bot.reset()
        # Rotate independent snapshots after the first chronological pass.
        order = list(range(len(corpus)))
        if repeat:
            random.Random(811223 + repeat).shuffle(order)
        for index in order:
            obs = corpus[index]
            start = perf_counter_ns()
            cpu_start = process_time_ns()
            action = bot.act(obs)
            cpu_elapsed = process_time_ns() - cpu_start
            elapsed = perf_counter_ns() - start
            wall.append(elapsed)
            cpu.append(cpu_elapsed)
            streets[obs["street"]].append(elapsed)
            illegal += not is_legal(obs, action)
            if elapsed > 1_000_000 and len(overruns) < 20:
                overruns.append({"index": index, "repeat": repeat, "street": obs["street"],
                                 "wall_us": elapsed / 1000, "cpu_us": cpu_elapsed / 1000,
                                 "history_events": len(obs["history"]), "opponents": obs["active_players"] - 1})
    report = {"label": "LOCAL latency; not official sandbox or hardware", "python": sys.version,
              "platform": platform.platform(), "gc_enabled": gc.isenabled(), "assets": flags,
              "initialization_us": initialization / 1000, "observations": len(corpus),
              "repeats": args.repeats, "wall": percentiles(wall), "process_cpu": percentiles(cpu),
              "street_wall": {str(s): percentiles(v) for s, v in streets.items()},
              "illegal": illegal, "internal_errors": bot.errors,
              "internal_budget_fallbacks": bot.budget_fallbacks, "sample_overruns": overruns,
              "cpu_min_positive_us": min((x for x in cpu if x > 0), default=0) / 1000,
              "cpu_submillisecond_resolution_observed": 0 < min((x for x in cpu if x > 0), default=0) < 1000000,
              "note": "Wall time includes scheduler interruptions and two timing calls. Repeated replays are latency measurements, not independent poker-strength samples."}
    if not report["cpu_submillisecond_resolution_observed"]:
        report["note"] += " The local process CPU clock is too coarse for per-decision interpretation; use wall times here and Linux CPU measurements separately."
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
