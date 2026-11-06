"""Reproducible LOCAL synthetic-field sit-and-go benchmark (no official API).

Paired seeds, randomized hero seat, 200-hand cap. Unknown official payouts and
blind levels are represented by explicitly named synthetic sensitivity cases.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokerbot.game import run_tournament
from pokerbot.opponents import Opponent
from pokerbot.strategy import Agent

PAYOUTS = dict(top3=(.5, .3, .2, 0, 0, 0, 0, 0), winner=(1., 0, 0, 0, 0, 0, 0, 0), linear=(8/36, 7/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36), flat3=(1/3, 1/3, 1/3, 0, 0, 0, 0, 0))
BLINDS = dict(moderate=((10, 20), (15, 30), (25, 50), (50, 100), (100, 200), (150, 300), (250, 500), (500, 1000)), fast=tuple((10 * 2**i, 20 * 2**i) for i in range(8)), slow=((10, 20), (15, 30), (20, 40), (30, 60), (40, 80), (60, 120), (80, 160), (120, 240)))
FIELDS = dict(mixed=("tag", "nit", "station", "lag", "maniac", "rollout", "balanced"), tight=("tag", "tag", "nit", "nit", "balanced", "rollout", "tag"), loose=("station", "station", "lag", "lag", "maniac", "rollout", "balanced"), aggressive=("lag", "lag", "maniac", "maniac", "tag", "rollout", "balanced"), rollout=("rollout",) * 7, self=("candidate",) * 7)


def play(task):
    seed, field, payouts, blinds, config, incumbent = task[:6]
    timeout = task[6] if len(task) > 6 else None
    rng = random.Random(seed)
    seat = rng.randrange(8)
    opponents = list(FIELDS[field])
    rng.shuffle(opponents)
    agents = []
    for i in range(8):
        if i == seat:
            agents.append(Agent(seed=seed * 101 + i, config=config))
        else:
            style = opponents.pop()
            agents.append(Agent(seed=seed * 101 + i, config=incumbent) if style == "candidate" else Opponent(style, seed=seed * 101 + i))
    r = run_tournament(agents, payouts=PAYOUTS[payouts], blind_schedule=BLINDS[blinds], seed=seed, strict=True, decision_timeout_ms=timeout)
    return dict(seed=seed, seat=seat, points=r.points[seat], finish=r.finishes[seat], chips=r.stacks[seat], hands=r.hands, decisions=r.decision_count, illegal=r.invalid_actions, exceptions=r.exceptions, timeouts=r.timeouts, budget_fallbacks=agents[seat].diagnostics["budget_fallbacks"])


def summary(rows):
    points = [r["points"] for r in rows]
    mean = statistics.mean(points)
    se = statistics.stdev(points) / len(points)**.5 if len(points) > 1 else 0
    return dict(tables=len(rows), mean_points=mean, se=se, ci95_normal=(mean - 1.96 * se, mean + 1.96 * se), relative_to_equal_share=mean / .125 - 1, wins=sum(r["finish"] == 1 for r in rows), mean_finish=statistics.mean(r["finish"] for r in rows), mean_hands=statistics.mean(r["hands"] for r in rows), decisions=sum(r["decisions"] for r in rows), illegal=sum(r["illegal"] for r in rows), exceptions=sum(r["exceptions"] for r in rows), timeouts=sum(r["timeouts"] for r in rows), budget_fallbacks=sum(r["budget_fallbacks"] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", type=int, default=100)
    p.add_argument("--seed", type=int, default=140000)
    p.add_argument("--field", choices=FIELDS, default="mixed")
    p.add_argument("--payouts", choices=PAYOUTS, default="top3")
    p.add_argument("--blinds", choices=BLINDS, default="moderate")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--config", default="{}", help="inline JSON or path to JSON config")
    p.add_argument("--incumbent", default="{}", help="self-field reference config")
    p.add_argument("--timeout-ms", type=float, default=None, help="enforce retrospective engine wall deadline")
    p.add_argument("--output", default="measurements/development.json")
    args = p.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.tables <= 20000:
        p.error("bounded workers1..8/tables1..20000 required")
    def load(x):
        return json.loads(x) if x.lstrip().startswith("{") else json.loads(Path(x).read_text())
    config, incumbent = load(args.config), load(args.incumbent)
    tasks = [(args.seed + i, args.field, args.payouts, args.blinds, config, incumbent, args.timeout_ms) for i in range(args.tables)]
    start = time.perf_counter()
    if args.workers == 1:
        rows = list(map(play, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(play, tasks, chunksize=10))
    report = dict(phase="development unless explicitly locked as held-out", synthetic_assumptions=True, timeout_ms=args.timeout_ms, field=args.field, payouts=args.payouts, blind_schedule=args.blinds, seed_start=args.seed, config=config, incumbent=incumbent, summary=summary(rows), seconds=time.perf_counter() - start, rows=rows)
    report["strategy_sha256"] = hashlib.sha256(Path(Agent.decide.__code__.co_filename).read_bytes()).hexdigest()
    report["asset_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__).resolve().parents[1] / "assets").glob("*.json") if "smoke" not in f.name}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
