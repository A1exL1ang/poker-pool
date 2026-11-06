"""Reproducible local evaluation against explicit synthetic opponent suites.

Run ``python -m sim.benchmark --help``. Results are local evidence only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from .baselines import Baseline
from .engine import SYNTHETIC_BLINDS, tournament

SUITES = {
    "mixed": ("calling", "nit", "tag", "lag", "maniac", "random", "pushfold"),
    "calling": ("calling",) * 7,
    "tight": ("nit", "tag", "nit", "tag", "pushfold", "nit", "tag"),
    "aggressive": ("lag", "maniac", "lag", "maniac", "tag", "random", "pushfold"),
    "balanced": ("tag",) * 7,
    # Reserved before evaluation: do not use this suite for parameter tuning.
    "challenge": ("range", "trap", "pressure", "range", "trap", "pressure", "range"),
    "selfplay": ("initial", "wide1.2", "select1.8", "aggressive1.6", "cautious", "noadapt", "loose"),
}
_INITIAL_CONFIG = {"open_scale": 1.0, "call_margin": .025, "aggression": 1.0,
                   "selection": 1.0, "shove_margin": .03, "adaptation": 1.0, "icm_shove":0.0}
SELFPLAY_CONFIGS = {
    "initial": dict(_INITIAL_CONFIG),
    "wide1.2": {**_INITIAL_CONFIG, "open_scale": 1.2},
    "select1.8": {**_INITIAL_CONFIG, "selection": 1.8},
    "aggressive1.6": {**_INITIAL_CONFIG, "aggression": 1.6},
    "cautious": {**_INITIAL_CONFIG, "open_scale": .85, "call_margin": .055, "shove_margin": .08},
    "noadapt": {**_INITIAL_CONFIG, "adaptation": 0.0},
    "loose": {**_INITIAL_CONFIG, "open_scale": 1.25, "call_margin": 0.0, "selection": .75},
}
PAYOUTS = {
    "top3": [50, 30, 20, 0, 0, 0, 0, 0],
    "winner": [100, 0, 0, 0, 0, 0, 0, 0],
    "linear": [35, 25, 18, 12, 6, 3, 1, 0],
    "flat4": [25, 25, 25, 25, 0, 0, 0, 0],
}


def source_hashes():
    """Hash only explicitly named files belonging to this bot's workspace."""
    root = Path(__file__).resolve().parent.parent
    paths = ("agent.py", "policy/icm.py", "policy/icm_fast.py", "policy/texture.py", "equity/model.py",
             "equity/preflop.py", "equity/features.py", "equity/evaluator.py",
             "equity/model.json", "equity/preflop.json", "equity/premium.json", "equity/special.py", "equity/compiled.py",
             "sim/engine.py", "sim/baselines.py", "sim/benchmark.py")
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in paths if (root / relative).is_file()}


def make_candidate(spec, seed, kwargs):
    if spec.startswith("baseline:"):
        return Baseline(spec.split(":", 1)[1], seed)
    module, name = spec.split(":", 1)
    cls = getattr(importlib.import_module(module), name)
    return cls(seed=seed, **kwargs)


def mean_ci(values):
    mean = statistics.mean(values) if values else 0
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0
    return {"n": len(values), "mean": mean, "standard_error": sem,
            "normal_95_ci": [mean - 1.96 * sem, mean + 1.96 * sem]}


def quantile(values, q):
    if not values:
        return 0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))]


def run(*, candidate="agent:Agent", kwargs=None, tables=100, seed=10000,
        suite="mixed", payout="top3", max_hands=200, enforce_timeout=False,
        expose_payouts=True, capture=None, capture_limit=10000, initial_stack=2000,
        blind_schedule=None):
    kwargs = kwargs or {}
    hashes = source_hashes()
    payouts = PAYOUTS[payout] if isinstance(payout, str) else payout
    rows, latencies = [], []
    captured = 0
    started = time.perf_counter()
    for index in range(tables):
        hero = index % 8
        table_seed = seed + index
        order = list(SUITES[suite])
        random.Random(table_seed ^ 0x28F91).shuffle(order)
        agents, opponent_labels = [], []
        for seat in range(8):
            agent_seed = table_seed * 17 + seat
            if seat == hero:
                agents.append(make_candidate(candidate, agent_seed, kwargs))
            else:
                label = order.pop()
                opponent_labels.append(label)
                if suite == "selfplay":
                    from agent import Agent
                    agents.append(Agent(seed=agent_seed, config=SELFPLAY_CONFIGS[label]))
                else:
                    agents.append(Baseline(label, seed=agent_seed))
        def capture_action(obs, action, elapsed, status):
            nonlocal captured
            if capture is not None and obs["hero"] == hero and captured < capture_limit:
                capture.write(json.dumps({"table_seed": table_seed, "observation": obs,
                                          "action": action, "latency_ns": elapsed, "status": status}) + "\n")
                captured += 1
        result = tournament(agents, seed=table_seed, initial_stack=initial_stack, max_hands=max_hands,
                            payouts=payouts, enforce_timeout=enforce_timeout,
                            blind_schedule=blind_schedule or SYNTHETIC_BLINDS,
                            capture_latencies=True, expose_payouts=expose_payouts,
                            decision_observer=capture_action if capture is not None else None)
        latencies.extend(result.latency_ns[hero])
        rows.append({"seed": table_seed, "hero": hero, "points": result.points[hero],
                     "stack": result.stacks[hero], "hands": result.hands,
                     "errors": result.errors[hero], "illegal": result.illegal[hero],
                     "candidate_fallbacks": getattr(agents[hero], "errors", 0),
                     "overruns": result.timeouts[hero], "decisions": result.decisions[hero],
                     "opponent_points": [v for i, v in enumerate(result.points) if i != hero],
                     "opponent_labels": opponent_labels,
                     "opponent_fallbacks": sum(getattr(a, "errors", 0) for i, a in enumerate(agents) if i != hero),
                     "opponent_errors": sum(result.errors) - result.errors[hero],
                     "opponent_illegal": sum(result.illegal) - result.illegal[hero]})
    ending_hashes = source_hashes()
    return {"kind": "LOCAL_SYNTHETIC_EVALUATION", "candidate": candidate,
            "candidate_kwargs": kwargs, "suite": suite, "payouts": payouts,
            "selfplay_opponent_configs": SELFPLAY_CONFIGS if suite == "selfplay" else None,
            "source_sha256": hashes,
            "source_changed_during_run": [p for p in set(hashes) | set(ending_hashes) if hashes.get(p) != ending_hashes.get(p)],
            "blind_schedule": blind_schedule or SYNTHETIC_BLINDS, "hands_per_level": 25,
            "max_hands": max_hands, "initial_stack": initial_stack,
            "seed_start": seed, "tables": tables, "timeout_enforced": enforce_timeout,
            "payouts_exposed_to_agent": expose_payouts,
            "points": mean_ci([r["points"] for r in rows]),
            "stack": mean_ci([r["stack"] for r in rows]),
            "errors": sum(r["errors"] for r in rows),
            "candidate_fallbacks": sum(r["candidate_fallbacks"] for r in rows),
            "opponent_fallbacks": sum(r["opponent_fallbacks"] for r in rows),
            "illegal": sum(r["illegal"] for r in rows),
            "decisions": len(latencies), "overruns": sum(r["overruns"] for r in rows),
            "latency_us": {"p50": quantile(latencies, .5) / 1000,
                           "p95": quantile(latencies, .95) / 1000,
                           "p99": quantile(latencies, .99) / 1000,
                           "p999": quantile(latencies, .999) / 1000,
                           "max": max(latencies, default=0) / 1000},
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": ["Synthetic payouts/blinds and heuristic opponents; no official compatibility claim.",
                            "Normal confidence intervals are approximate; tables are independent units.",
                            "Wall-clock timing includes scheduler pauses; initialization is excluded.",
                            "Same-seed comparisons share initial decks but subsequent paths can diverge."],
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="agent:Agent")
    parser.add_argument("--kwargs", default="{}", help="JSON constructor keyword arguments, excluding seed")
    parser.add_argument("--compare-kwargs", help="Optional paired second candidate parameters as JSON")
    parser.add_argument("--tables", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--suite", choices=SUITES, default="mixed")
    parser.add_argument("--payout", choices=PAYOUTS, default="top3", help="Synthetic points, not official payouts")
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument("--enforce-timeout", action="store_true")
    parser.add_argument("--hide-payouts", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--capture", type=Path, help="Optional hero-observation JSONL for diagnostics")
    parser.add_argument("--capture-limit", type=int, default=10000)
    args = parser.parse_args()
    capture = None
    if args.capture:
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        capture = args.capture.open("w", encoding="utf-8")
    common = {"candidate": args.candidate, "tables": args.tables, "seed": args.seed,
              "suite": args.suite, "payout": args.payout, "max_hands": args.max_hands,
              "enforce_timeout": args.enforce_timeout, "expose_payouts": not args.hide_payouts}
    try:
        result = run(**common, kwargs=json.loads(args.kwargs), capture=capture, capture_limit=args.capture_limit)
    finally:
        if capture is not None:
            capture.close()
    if args.compare_kwargs:
        other = run(**common, kwargs=json.loads(args.compare_kwargs))
        differences = [a["points"] - b["points"] for a, b in zip(result["rows"], other["rows"])]
        result["paired_difference_primary_minus_comparison"] = mean_ci(differences)
        result["comparison"] = other
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "comparison")}, indent=2))


if __name__ == "__main__":
    main()
