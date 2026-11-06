"""Reproducible LOCAL performance experiment with optional paired control.

Example (hypothetical blind/payout assumptions):
  python -m simulator.benchmark --candidate agent:Agent --tables 64 \
      --field mixed --blinds medium --payout top3 --split train \
      --reference model_equity --output measurements/train_smoke.json

Large jobs should be run only on this bot's own bounded compute allocation.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
from math import sqrt
from pathlib import Path
import statistics
import sys
from time import perf_counter

from .baselines import BASELINES, make_baseline
from .engine import play_tournament
from .scenarios import BLINDS, FIELDS, PAYOUTS


SEED_START = {"train": 10000, "validation": 5000000, "heldout": 90000000}


def capture_provenance(options):
    """Hash this bot's source/assets once at experiment launch, never in act().

    Scope is our local Agent, pokerbot and simulator Python sources, explicit
    custom policy entry modules, every packaged pokerbot asset, and the frozen
    experiment protocol when present. No parent/sibling directory is inspected.
    Files must remain unchanged throughout an experiment.
    """
    root = Path(__file__).resolve().parents[1]
    sources = {root / "agent.py"}
    for directory in (root / "pokerbot", root / "simulator"):
        if directory.is_dir():
            sources.update(directory.rglob("*.py"))
    specs = [options["candidate"], options.get("reference"), *options["opponents"]]
    for spec in specs:
        if spec and ":" in spec:
            module_path = root.joinpath(*spec.split(":", 1)[0].split("."))
            for path in (module_path.with_suffix(".py"), module_path / "__init__.py"):
                if path.is_file():
                    sources.add(path)
    assets_dir = root / "pokerbot" / "assets"
    assets = set(assets_dir.rglob("*")) if assets_dir.is_dir() else set()
    protocol = root / "benchmarks" / "protocol.json"
    files = {}
    for kind, paths in (("source", sources), ("asset", assets), ("protocol", {protocol})):
        for path in sorted(paths):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("provenance file resolves outside this bot workspace")
            if not path.is_file():
                continue
            data = path.read_bytes()
            files[path.relative_to(root).as_posix()] = {
                "kind": kind, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "SHA256", "fingerprint_sha256": fingerprint, "files": files,
            "capture_policy": "Captured once before experiment; freeze source/assets for the full run."}


def model_flags(policy):
    estimator = getattr(policy, "estimator", None)
    if estimator is None:
        return {"exposes_estimator": False}
    return {"exposes_estimator": True,
            "model_loaded": bool(getattr(estimator, "model_loaded", False)),
            "preflop_loaded": bool(getattr(estimator, "preflop_loaded", False))}


def require_loaded_models(policy, label, allow_fallback=False):
    flags = model_flags(policy)
    if (flags["exposes_estimator"] and not allow_fallback
            and not (flags["model_loaded"] and flags["preflop_loaded"])):
        raise ValueError(f"{label} has missing/incompatible model assets: {flags}. "
                         "For an explicitly labeled smoke test only, use --allow-fallback-model.")
    return flags


def preflight_models(options):
    """Check each distinct candidate/control/opponent before starting tables."""
    specs = [("candidate", options["candidate"], options["candidate_config"])]
    if options["reference"]:
        specs.append(("reference", options["reference"], options["reference_config"]))
    specs.extend(("opponent", spec, options["opponent_config"] if ":" in spec else {})
                 for spec in options["opponents"])
    unique = {}
    for role, spec, config in specs:
        key = (spec, json.dumps(config, sort_keys=True))
        if key in unique:
            unique[key]["roles"] = sorted(set(unique[key]["roles"] + [role]))
            continue
        policy = load_policy(spec, options["seed_start"], config)
        flags = require_loaded_models(policy, f"{role} {spec}", options["allow_fallback_model"])
        unique[key] = {"roles": [role], "policy": spec, "config": config, "flags": flags}
    return list(unique.values())


def load_policy(spec, seed, config=None):
    """Build our own local classes; generic imports are explicit CLI inputs."""
    if spec in BASELINES:
        if config:
            raise ValueError("baseline policy constructors do not accept candidate config")
        return make_baseline(spec, seed)
    module_name, name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, name)
    parameters = inspect.signature(factory).parameters
    kwargs = {"seed": seed} if "seed" in parameters else {}
    if config:
        if "config" in parameters:
            config_factory = getattr(module, "PolicyConfig", None)
            kwargs["config"] = config_factory(**config) if config_factory else dict(config)
        else:
            kwargs.update(config)
    return factory(**kwargs)


def _one_table(task):
    index, options = task
    seed = options["seed_start"] + (index // 8 if options["seating"] == "rotations" else index)
    hero = index % 8
    names = options["opponents"]

    def run(candidate, candidate_config):
        agents = []
        cursor = 0
        for seat in range(8):
            agent_seed = (seed * 1009 + seat * 9176 + 20) & 0xFFFFFFFF
            if seat == hero:
                agents.append(load_policy(candidate, agent_seed, candidate_config))
            else:
                name = names[cursor]
                config = options["opponent_config"] if ":" in name else None
                agents.append(load_policy(name, agent_seed, config))
                cursor += 1
            require_loaded_models(agents[-1], f"seat {seat}", options.get("allow_fallback_model", False))
        result = play_tournament(agents, seed=seed, max_hands=options["max_hands"],
                                 blind_levels=BLINDS[options["blinds"]],
                                 payouts=PAYOUTS[options["payout"]], strict=True,
                                 enforce_time=options["enforce_time"])
        candidate_agent = agents[hero]
        internal = {name: getattr(candidate_agent, name) for name in
                    ("errors", "budget_fallbacks", "decisions") if hasattr(candidate_agent, name)}
        return {"score": result["scores"][hero], "chips": result["stacks"][hero],
                "won": result["winner"] == hero, "cutoff": result["cutoff"],
                "hands": result["hands"], "metrics": result["metrics"][hero],
                "internal": internal, "model_flags": model_flags(candidate_agent)}

    row = {"index": index, "seed": seed, "hero": hero}
    row.update(run(options["candidate"], options["candidate_config"]))
    if options["reference"]:
        reference = run(options["reference"], options["reference_config"])
        row["reference"] = reference
        row["paired_delta"] = row["score"] - reference["score"]
    return row


def _mean_ci(values):
    n = len(values)
    mean = statistics.fmean(values) if values else 0.0
    se = statistics.stdev(values) / sqrt(n) if n > 1 else None
    return {"mean": mean, "standard_error": se,
            "approx_95pct_ci": [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
            "independent_seed_blocks": n}


def summarize(rows, seating):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["seed"], []).append(row)
    scores = [statistics.fmean(r["score"] for r in group) for group in grouped.values()]
    result = {"score": _mean_ci(scores), "tables": len(rows),
              "outright_win_rate": statistics.fmean(r["won"] for r in rows),
              "cutoff_rate": statistics.fmean(r["cutoff"] for r in rows),
              "average_hands": statistics.fmean(r["hands"] for r in rows)}
    result["score"]["equal_seat_share_reference"] = 1 / 8
    if "paired_delta" in rows[0]:
        differences = [statistics.fmean(r["paired_delta"] for r in group) for group in grouped.values()]
        reference = [statistics.fmean(r["reference"]["score"] for r in group) for group in grouped.values()]
        result["paired_delta"] = _mean_ci(differences)
        result["reference_score"] = _mean_ci(reference)
    metrics = {}
    for name in ("decisions", "errors", "illegal", "over_1ms", "fallbacks", "elapsed_ns"):
        metrics[name] = sum(r["metrics"][name] for r in rows)
    metrics["max_ns"] = max(r["metrics"]["max_ns"] for r in rows)
    metrics["mean_us"] = metrics["elapsed_ns"] / max(1, metrics["decisions"]) / 1000
    result["candidate_metrics"] = metrics
    internal_keys = {key for row in rows for key in row["internal"]}
    result["candidate_internal"] = {key: sum(row["internal"].get(key, 0) for row in rows) for key in internal_keys}
    return result


def _parse_config(value):
    if not value:
        return {}
    content = Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value
    config = json.loads(content)
    if not isinstance(config, dict):
        raise ValueError("policy config must be a JSON object")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="agent:Agent")
    parser.add_argument("--candidate-config", default="{}", help="JSON object or @local-file.json")
    parser.add_argument("--reference", default=None, help="matched control baseline or local module:class")
    parser.add_argument("--reference-config", default="{}")
    parser.add_argument("--opponent-config", default="{}", help="applied only to imported local opponent classes")
    parser.add_argument("--field", choices=sorted(FIELDS), default="mixed")
    parser.add_argument("--opponents", help="seven comma-separated baseline or local module:class specs")
    parser.add_argument("--blinds", choices=sorted(BLINDS), default="medium")
    parser.add_argument("--payout", choices=sorted(PAYOUTS), default="top3")
    parser.add_argument("--tables", type=int, default=64)
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--split", choices=tuple(SEED_START), default="train")
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seating", choices=("independent", "rotations"), default="independent")
    parser.add_argument("--enforce-time", action="store_true")
    parser.add_argument("--allow-fallback-model", action="store_true",
                        help="explicitly permit missing/incompatible trained assets for smoke tests")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    if args.tables < 1 or not 1 <= args.workers <= 8:
        parser.error("tables must be positive; workers must be 1..8")
    if args.seating == "rotations" and args.tables % 8:
        parser.error("rotation evaluation needs tables divisible by eight")
    if not 1 <= args.max_hands <= 200:
        parser.error("max-hands must be 1..200")
    options = vars(args).copy()
    options["seed_start"] = SEED_START[args.split] if args.seed_start is None else args.seed_start
    options["candidate_config"] = _parse_config(args.candidate_config)
    options["reference_config"] = _parse_config(args.reference_config)
    options["opponent_config"] = _parse_config(args.opponent_config)
    options["opponents"] = args.opponents.split(",") if args.opponents else FIELDS[args.field]
    if len(options["opponents"]) != 7:
        parser.error("exactly seven opponent specs required")
    provenance = capture_provenance(options)
    model_preflight = preflight_models(options)
    started = perf_counter()
    tasks = [(i, options) for i in range(args.tables)]
    if args.workers == 1:
        rows = [_one_table(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_one_table, tasks, chunksize=max(1, args.tables // (args.workers * 8))))
    result = {
        "label": "Independent LOCAL experiment; blind progression and payouts are hypothetical",
        "confidence_note": "Normal 95% intervals over independent deck-seed blocks; descriptive, not proof against an unknown field. Tuning data must not be reported as held-out.",
        "options": options, "blind_levels": BLINDS[args.blinds], "payouts": PAYOUTS[args.payout],
        "provenance": provenance, "model_preflight": model_preflight,
        "summary": summarize(rows, args.seating), "elapsed_seconds": perf_counter() - started,
        "python": sys.version, "rows": rows,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "elapsed_seconds": result["elapsed_seconds"],
                      "output": args.output}, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
