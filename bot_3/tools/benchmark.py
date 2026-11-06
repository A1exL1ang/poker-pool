"""Reproducible local latency and tournament evidence; no official compatibility."""
import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

from bot3.agent import Agent
from bot3.equity import EquityModel
from bot3.interface import validate_action, validate_observation
from tools.baselines import ScriptedOpponent
from tools.simulator import Table


def quantile(values, q):
    return sorted(values)[min(len(values) - 1, int(q * len(values)))]


def latency(model, count, seed):
    rng = random.Random(seed)
    a = Agent(seed=seed, model=model)
    observations = []
    for k in range(max(200, count // 10)):
        n = rng.randrange(2, 9)
        hero = rng.randrange(n)
        villain = (hero + 1) % n
        nb = rng.choice((0, 3, 4, 5))
        cs = rng.sample(range(52), nb + 2)
        ps = [{"seat": i, "stack": rng.randrange(100, 15000), "bet": 0, "committed": 0, "folded": False, "all_in": False} for i in range(n)]
        bet = rng.randrange(0, 5000)
        ps[villain].update(bet=bet, committed=bet + 400)
        call = min(ps[hero]["stack"], bet)
        maximum = ps[hero]["stack"]
        legal = (["check"] if not call else ["fold", "call"]) + (["raise"] if maximum > bet else [])
        o = {"table_id": "latency", "hand_id": k, "hero_seat": hero, "button": rng.randrange(n), "hole_cards": cs[:2], "board": cs[2:],
             "players": ps, "pot": bet + rng.randrange(100, 4000), "big_blind": 100, "to_call": call,
             "street": {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[nb], "min_raise_to": bet + max(100, bet), "max_raise_to": maximum,
             "legal_actions": legal, "history": []}
        # Exercise the most expensive optional ICM path in the heads-up cases.
        if n == 2 and k % 2:
            o["payouts"] = [5, 3, 2]
        validate_observation(o)
        observations.append(o)
    for o in observations:
        validate_action(o, a.act(o))
    times, cpu_times = [], []
    by_street = {s: [] for s in ("preflop", "flop", "turn", "river")}
    for k in range(count):
        o = observations[k % len(observations)]
        cpu_start = time.thread_time_ns()
        start = time.perf_counter_ns()
        result = a.act(o)
        ns = time.perf_counter_ns() - start
        cpu_ns = time.thread_time_ns() - cpu_start
        validate_action(o, result)
        times.append(ns / 1000)
        cpu_times.append(cpu_ns / 1000)
        by_street[o["street"]].append(ns / 1000)
    def summary(ts):
        return {"count": len(ts), "mean_us": statistics.mean(ts), "p50_us": quantile(ts, .5), "p95_us": quantile(ts, .95), "p99_us": quantile(ts, .99), "p999_us": quantile(ts, .999), "max_us": max(ts), "over_1ms": sum(x > 1000 for x in ts)}
    cpu = {"total_ms": sum(cpu_times) / 1000, "mean_us": statistics.mean(cpu_times)}
    if platform.system() == "Windows":
        cpu["note"] = "Windows CPU accounting is quantized; per-decision CPU percentiles are not meaningful on this host."
    else:
        cpu.update(summary(cpu_times))
        cpu["note"] = "Thread CPU includes clock-call overhead; wall latency remains the relevant strict deadline measure."
    return {"all": summary(times), "thread_cpu": cpu,
            "streets": {s: summary(t) for s, t in by_street.items()}, "timer": "perf_counter_ns wall and aggregate thread_time_ns CPU; act only; constructor, JSON, validation excluded"}


def tournaments(model, count, seed, suite, deadline_us=1000):
    profiles = {
        "mixed": ["balanced", "pressure", "tight", "passive", "random", "balanced", "pressure"],
        "balanced": ["balanced"] * 7,
        "pressure": ["pressure"] * 7,
        "passive": ["passive"] * 7,
        "tight": ["tight"] * 7,
    }
    styles = profiles[suite]
    shares, wins, rank_points, hands = [], 0, [], []
    ac = {}
    hero_timeouts, opponent_timeouts = 0, 0
    t0 = time.perf_counter()
    for k in range(count):
        seat = k % 8
        hero = Agent(seed=seed + k * 17, model=model)
        agents = [ScriptedOpponent(model, s, seed + k * 31 + j) for j, s in enumerate(styles)]
        agents.insert(seat, hero)
        table = Table(agents, seed=seed + k * 1009, decision_budget_ns=deadline_us * 1000 if deadline_us else None)
        result = table.run()
        stacks = result["stacks"]
        shares.append(stacks[seat] / 16000)
        wins += stacks[seat] == 16000
        hands.append(result["hands"])
        hero_timeouts += result["timeouts"][seat]
        opponent_timeouts += sum(result["timeouts"]) - result["timeouts"][seat]
        # Rank metric orders eliminated players; survivors use remaining stacks.
        groups = list(result["elimination_groups"])
        for value in sorted({s for s in stacks if s > 0}):
            groups.append([i for i, s in enumerate(stacks) if s == value])
        passed = 0
        for group in groups:
            if seat in group:
                rank_points.append(passed + (len(group) + 1) / 2)
                break
            passed += len(group)
        for action, n in result["actions"][seat].items():
            ac[action] = ac.get(action, 0) + n
        if (k + 1) % 50 == 0:
            print(f"{suite}: {k+1}/{count}, mean terminal chip share {statistics.mean(shares):.4f}", file=sys.stderr, flush=True)
    mean = statistics.mean(shares)
    se = statistics.stdev(shares) / math.sqrt(count) if count > 1 else 0
    return {"suite": suite, "tables": count, "seed": seed, "hero_seat": "cycled 0..7", "first_place_finishes": wins,
            "mean_terminal_chip_share": mean, "standard_error": se, "approx_95pct_ci": [mean - 1.96 * se, mean + 1.96 * se],
            "equal_share_reference": .125, "mean_finish_order_low_to_high": statistics.mean(rank_points), "mean_finish_place": 9 - statistics.mean(rank_points), "mean_hands": statistics.mean(hands),
            "hero_actions": ac, "seconds": time.perf_counter() - t0,
            "deadline_us": deadline_us, "hero_timeout_fallbacks": hero_timeouts, "opponent_timeout_fallbacks": opponent_timeouts,
            "assumptions": "synthetic scripted opponents; 2000 chips; 10/20 initial blinds doubled every 25 hands; 200-hand cap; no supplied payouts; not a contest score"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latency", type=int, default=0)
    p.add_argument("--tables", type=int, default=0)
    p.add_argument("--suite", choices=("mixed", "balanced", "pressure", "passive", "tight"), default="mixed")
    p.add_argument("--seed", type=int, default=20261106)
    p.add_argument("--deadline-us", type=int, default=1000, help="tournament decision limit; 0 disables timeout fallback")
    p.add_argument("--output", default="evidence/benchmark.json")
    args = p.parse_args()
    t0 = time.perf_counter()
    model = EquityModel()
    output = {"python": sys.version, "platform": platform.platform(), "model_load_ms": (time.perf_counter() - t0) * 1000, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path("bot3").glob("*.py"), Path("assets/preflop.json"), Path("assets/equity_model.json"), Path("assets/pushfold.json")]}}
    if args.latency:
        output["latency"] = latency(model, args.latency, args.seed)
    if args.tables:
        output["tournaments"] = tournaments(model, args.tables, args.seed, args.suite, args.deadline_us)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
