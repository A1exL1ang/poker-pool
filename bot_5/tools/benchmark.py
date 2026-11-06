"""Measure complete Agent.act calls on held-out, legal game observations.

Asset loading and simulator bookkeeping are outside the timed region.
Wall-clock tails are reported honestly, including scheduler/GC interruptions.
"""
import argparse
from collections import Counter, defaultdict
import gc
import json
import platform
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pokerbot.agent import Agent
from pokerbot.equity import Equity
from pokerbot.icm import icm_equity
from simulator import Baseline, Table, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=81000)
    parser.add_argument("--output", default="evidence/latency.json")
    parser.add_argument("--save-corpus", action="store_true")
    args = parser.parse_args()
    observations = []
    asset = Equity()
    load_started = time.perf_counter_ns()
    bot = Agent()
    load_us = (time.perf_counter_ns() - load_started) / 1000
    for t in range(args.tables):
        hero = t % 8
        seed = args.seed + t
        bots = [Baseline(("pressure", "tight", "station", "random", "pressure", "station", "tight")[i], seed * 13 + i) for i in range(7)]
        bots.insert(hero, bot)
        table = Table(bots, seed=seed, payouts=(.5, .3, .2, 0, 0, 0, 0, 0), capture=lambda obs, action, elapsed, hero=hero: observations.append(obs) if obs["hero"] == hero else None)
        table.run()
    if args.save_corpus:
        with (ROOT / "evidence" / "benchmark_corpus.jsonl").open("w") as f:
            for obs in observations:
                f.write(json.dumps(obs, separators=(",", ":")) + "\n")
    wall, cpu = [], []
    by_street = defaultdict(list)
    by_newhand = defaultdict(list)
    slow = []
    illegal = 0
    fallback = 0
    for repeat in range(args.repeats):
        icm_equity.cache_clear()
        bot.reset()
        gc.collect()
        for index, obs in enumerate(observations):
            new_hand = obs["table_id"] != bot.table_id or obs["hand_id"] != bot.hand_id
            c0 = time.thread_time_ns()
            t0 = time.perf_counter_ns()
            action = bot.act(obs)
            elapsed = (time.perf_counter_ns() - t0) / 1000
            cpu_elapsed = (time.thread_time_ns() - c0) / 1000
            wall.append(elapsed)
            cpu.append(cpu_elapsed)
            by_street[obs["street"]].append(elapsed)
            by_newhand["first_decision_of_hand" if new_hand else "later_decision"].append(elapsed)
            legal = obs["legal_actions"]
            kind = action["action"]
            valid = kind in legal
            if kind == "raise":
                valid = valid and legal["raise"]["min"] <= action["amount"] <= legal["raise"]["max"]
            elif kind == "call":
                valid = valid and legal["call"] == action["amount"]
            illegal += not valid
            fallback += bot.last_error is not None
            if elapsed > 1000:
                slow.append({"repeat": repeat, "index": index, "street": obs["street"], "new_hand": new_hand, "wall_us": elapsed, "thread_cpu_us": cpu_elapsed})
    # JSON decode + dispatch encode, measured separately from the Python API.
    lines = [json.dumps(obs, separators=(",", ":")) for obs in observations[:1000]]
    transport = []
    bot.reset()
    for line in lines:
        t0 = time.perf_counter_ns()
        json.dumps(bot.act(json.loads(line)), separators=(",", ":"))
        transport.append((time.perf_counter_ns() - t0) / 1000)
    result = {
        "settings": vars(args), "python": sys.version, "platform": platform.platform(),
        "description": "Complete Agent.act including validation, history tracking, optional ICM and decision. Init, JSON, I/O excluded from main measurement. Thread CPU timer may be coarse on Windows; wall-clock is authoritative here.",
        "asset_load_us": load_us, "unique_legal_observations": len(observations),
        "wall_us": summarize(wall), "thread_cpu_us": summarize(cpu),
        "by_street_wall_us": {k: summarize(v) for k, v in by_street.items()},
        "by_hand_position_wall_us": {k: summarize(v) for k, v in by_newhand.items()},
        "json_decode_act_encode_wall_us": summarize(transport),
        "illegal_actions": illegal, "fallback_errors": fallback,
        "slow_calls": sorted(slow, key=lambda x: x["wall_us"], reverse=True)[:50],
        "warning": "Not an official sandbox timing guarantee. Raw wall-clock outliers above 1 ms remain visible and may cause folds under the competition's rule.",
    }
    (ROOT / args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "slow_calls"}, indent=2))
    if illegal or fallback:
        raise SystemExit("invalid action or policy fallback in benchmark")


if __name__ == "__main__":
    main()
