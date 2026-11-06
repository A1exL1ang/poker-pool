"""Reproducible local tournament comparisons and decision timing.

Examples (every payout/schedule below is a LOCAL scenario, not official):
  python -m lab.benchmark --tables 64 --seed-start 1000 --payouts '[1,0,0,0,0,0,0,0]' --output measurements/tune.json
  python -m lab.benchmark --tables 64 --seed-start 900000 --lineup tight --payouts '[50,30,20,0,0,0,0,0]' --output measurements/heldout.json
  python -m lab.benchmark --tables 64 --opponents reference,reference,reference,reference,reference,reference,reference --reference-config '{"adapt":false}' --output measurements/ablation.json

Distinct seeds are used per table. Candidate seat cycles through all seats.
Running another configuration with the same arguments supplies paired seeds;
strategic actions can of course change the subsequent trajectory/deal usage.
The caller must keep tuning seeds separate from held-out evaluation seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter

from lab.baselines import BASELINES, make_baseline
from lab.simulator import Tournament


LOCAL_BLIND_SCENARIOS = {
    "gentle": [(10, 20), (15, 30), (25, 50), (40, 80), (60, 120), (100, 200), (150, 300), (250, 500)],
    "doubling": [(10 * 2**i, 20 * 2**i) for i in range(8)],
    "fixed": [(10, 20)],
}
LINEUPS = {
    "mixed": ["tight", "station", "aggressive", "equity", "tight", "station", "aggressive"],
    "tight": ["tight"] * 7,
    "station": ["station"] * 7,
    "aggressive": ["aggressive"] * 7,
    "equity": ["equity"] * 7,
    "random": ["random"] * 7,
    "passive": ["checkfold"] * 7,
    "shovewide": ["shovewide"] * 7,
    "premiumnit": ["premiumnit"] * 7,
    "minpressure": ["minpressure"] * 7,
    "stress": ["shovewide", "premiumnit", "minpressure", "shovewide", "premiumnit", "minpressure", "equity"],
}


def summarize(values):
    if not values:
        return {"count": 0, "mean": None, "standard_error": None, "ci95_normal": None}
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return {
        "count": len(values), "mean": mean, "standard_error": se,
        "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
    }


def latency_summary(records):
    times = sorted(r.elapsed_ns for r in records)
    def quantile(p):
        return times[min(len(times) - 1, max(0, math.ceil(p * len(times)) - 1))] / 1000 if times else None
    return {
        "count": len(times),
        "mean_us": statistics.fmean(times) / 1000 if times else None,
        "median_us": quantile(0.5), "p95_us": quantile(0.95),
        "p99_us": quantile(0.99), "max_us": times[-1] / 1000 if times else None,
        "over_1ms": sum(t > 1_000_000 for t in times),
        "fallbacks": dict(Counter(r.fallback for r in records if r.fallback)),
        "errors": dict(Counter(r.error for r in records if r.error)),
    }


def import_policy(spec):
    module, sep, name = spec.partition(":")
    if not sep:
        raise ValueError("policy spec must have the form module:Class")
    return getattr(importlib.import_module(module), name)


def construct(factory, seed, config):
    params = inspect.signature(factory).parameters
    kwargs = {}
    if "seed" in params:
        kwargs["seed"] = seed
    if "config" in params:
        kwargs["config"] = config
    elif config:
        raise ValueError("selected policy constructor does not accept config")
    return factory(**kwargs)


class TracePolicy:
    def __init__(self, policy, observations):
        self.policy = policy
        self.observations = observations

    def act(self, obs):
        decision = self.policy.act(obs)
        self.observations.append({"observation": obs, "decision": decision})
        return decision


def run_benchmark(
    *, candidate="bot13.agent:Agent", config=None, reference=None, reference_config=None,
    opponents=None, tables=64, seed_start=1000, blind_schedule=None,
    payouts=None, hand_limit=200, hands_per_level=25, starting_stack=2000,
    enforce_deadline=False, observations_path=None, progress_every=8,
):
    if tables < 1:
        raise ValueError("tables must be positive")
    config = config or {}
    opponents = opponents or LINEUPS["mixed"]
    if len(opponents) != 7:
        raise ValueError("exactly seven opponent names are required")
    if any(name not in BASELINES and name != "reference" for name in opponents):
        raise ValueError("unknown opponent name")
    schedule = blind_schedule or LOCAL_BLIND_SCENARIOS["gentle"]
    factory = import_policy(candidate)
    ref_factory = import_policy(reference or candidate) if "reference" in opponents else None
    ref_config = reference_config if reference_config is not None else config
    table_results = []
    candidate_records = []
    candidate_counters = Counter()
    all_records = []
    observed = []
    obs_handle = None
    if observations_path:
        obs_path = Path(observations_path)
        obs_path.parent.mkdir(parents=True, exist_ok=True)
        obs_handle = gzip.open(obs_path, "wt", encoding="utf-8") if obs_path.suffix == ".gz" else obs_path.open("w", encoding="utf-8")
    start = perf_counter()
    try:
        for table_id in range(tables):
            seed = seed_start + table_id
            hero = table_id % 8
            agents = []
            opponent_index = 0
            for seat in range(8):
                policy_seed = seed * 17 + seat
                if seat == hero:
                    policy = construct(factory, policy_seed, config)
                    agents.append(TracePolicy(policy, observed) if obs_handle else policy)
                else:
                    name = opponents[opponent_index]
                    opponent_index += 1
                    agents.append(construct(ref_factory, policy_seed, ref_config) if name == "reference" else make_baseline(name, policy_seed))
            result = Tournament(
                agents, blind_schedule=schedule, stacks=starting_stack, payouts=payouts,
                hands_per_level=hands_per_level, hand_limit=hand_limit,
                seed=seed, button=seed % 8, enforce_deadline=enforce_deadline,
            ).run()
            records = [r for r in result.decisions if r.seat == hero]
            candidate_records.extend(records)
            all_records.extend(result.decisions)
            candidate_policy = agents[hero].policy if isinstance(agents[hero], TracePolicy) else agents[hero]
            internal = {key: getattr(candidate_policy, key, 0) for key in ("decisions", "fallbacks", "native_errors", "native_calls")}
            candidate_counters.update(internal)
            candidate_counters["tables_with_native"] += int(getattr(candidate_policy, "native", None) is not None)
            record = {
                "table_id": table_id, "seed": seed, "candidate_seat": hero,
                "hands": result.hands, "stacks": result.stacks, "points": result.points,
                "candidate_chips": result.stacks[hero],
                "candidate_chip_delta": result.stacks[hero] - starting_stack,
                "candidate_chip_share": result.stacks[hero] / (starting_stack * 8),
                "candidate_points": result.points[hero] if result.points is not None else None,
                "winner": result.winner, "candidate_won_table": result.winner == hero,
                "candidate_latency": latency_summary(records),
                "candidate_internal": internal,
                "all_fallbacks": dict(Counter(r.fallback for r in result.decisions if r.fallback)),
                "eliminations": result.eliminations,
            }
            table_results.append(record)
            if obs_handle:
                for observation in observed:
                    observation.update({"table_id": table_id, "seed": seed})
                    obs_handle.write(json.dumps(observation, separators=(",", ":")) + "\n")
                observed.clear()
                obs_handle.flush()
            if progress_every and ((table_id + 1) % progress_every == 0 or table_id + 1 == tables):
                average = statistics.fmean(r["candidate_chips"] for r in table_results)
                print(f"{table_id + 1}/{tables} tables; candidate mean chips {average:.1f}; {perf_counter() - start:.1f}s", file=sys.stderr, flush=True)
    finally:
        if obs_handle:
            obs_handle.close()
    hashes = {}
    root = Path(__file__).resolve().parent.parent
    for relative in (
        "lab/simulator.py", "lab/baselines.py", "lab/benchmark.py", "bot13/agent.py",
        "bot13/native.py", "bot13/cards.py", "bot13/icm.py", "bot13/interface.py",
        "native/bot13.cpp", "assets/preflop.json", "assets/pushfold.json", "assets/policy.json",
    ):
        path = root / relative
        if path.is_file():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "independent_local_simulator": True,
            "official_interface_compatibility_claimed": False,
            "local_scenario_blinds_and_payouts_not_official": True,
            "candidate": candidate, "config": config,
            "reference": reference or candidate, "reference_config": ref_config,
            "opponents": opponents, "tables": tables, "seed_start": seed_start,
            "candidate_seat_rotation": "table_id modulo 8",
            "blind_schedule": schedule, "payouts": payouts,
            "hand_limit": hand_limit, "hands_per_level": hands_per_level,
            "starting_stack": starting_stack, "enforce_1ms_deadline": enforce_deadline,
            "observations_path": str(observations_path) if observations_path else None,
            "elapsed_seconds": perf_counter() - start,
            "source_sha256": hashes,
            "confidence_interval": "unadjusted normal 95% interval of independent seeded tables; small batches are noisy",
        },
        "summary": {
            "candidate_chips": summarize([r["candidate_chips"] for r in table_results]),
            "candidate_chip_delta": summarize([r["candidate_chip_delta"] for r in table_results]),
            "candidate_chip_share": summarize([r["candidate_chip_share"] for r in table_results]),
            "candidate_points": summarize([r["candidate_points"] for r in table_results]) if payouts is not None else None,
            "tables_won": sum(r["candidate_won_table"] for r in table_results),
            "tables_unfinished_at_hand_limit": sum(r["winner"] is None for r in table_results),
            "candidate_latency": latency_summary(candidate_records),
            "candidate_internal": dict(candidate_counters),
            "all_policies_latency": latency_summary(all_records),
        },
        "tables": table_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", default="bot13.agent:Agent")
    parser.add_argument("--config", default="{}", help="JSON constructor config")
    parser.add_argument("--reference", default=None, help="optional reference policy module:Class")
    parser.add_argument("--reference-config", default=None, help="JSON constructor config for reference opponents")
    parser.add_argument("--lineup", choices=sorted(LINEUPS), default="mixed")
    parser.add_argument("--opponents", default=None, help="seven comma-separated baseline names or reference; overrides --lineup")
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--blind-profile", choices=sorted(LOCAL_BLIND_SCENARIOS), default="gentle")
    parser.add_argument("--blind-schedule", default=None, help="explicit JSON list of [small,big] blind levels")
    parser.add_argument("--payouts", default=None, help="explicit JSON first-to-last LOCAL payout vector; none means chips only")
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--hands-per-level", type=int, default=25)
    parser.add_argument("--starting-stack", type=int, default=2000)
    parser.add_argument("--enforce-deadline", action="store_true")
    parser.add_argument("--observations", default=None, help="optional JSONL(.gz) candidate observation/action corpus")
    parser.add_argument("--output", default=None)
    parser.add_argument("--progress-every", type=int, default=8)
    args = parser.parse_args()
    result = run_benchmark(
        candidate=args.candidate, config=json.loads(args.config),
        reference=args.reference,
        reference_config=json.loads(args.reference_config) if args.reference_config is not None else None,
        opponents=args.opponents.split(",") if args.opponents else LINEUPS[args.lineup],
        tables=args.tables, seed_start=args.seed_start,
        blind_schedule=json.loads(args.blind_schedule) if args.blind_schedule else LOCAL_BLIND_SCENARIOS[args.blind_profile],
        payouts=json.loads(args.payouts) if args.payouts else None,
        hand_limit=args.hands, hands_per_level=args.hands_per_level,
        starting_stack=args.starting_stack, enforce_deadline=args.enforce_deadline,
        observations_path=args.observations, progress_every=args.progress_every,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
