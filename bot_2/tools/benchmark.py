"""Reproducible local performance evidence. All opponents and rules are local."""

import argparse
import hashlib
import gc
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from pokerbot import Agent, Observation, Player, Event
from pokerbot.cards import evaluate
from tools.baselines import CallingStation, CheckFold, Shove, TightAggressive
from tools.simulator import Table


def quantiles(values):
    s = sorted(values)
    return {"count": len(s), "mean_us": statistics.mean(s),
            "p50_us": s[len(s) // 2], "p95_us": s[int(len(s) * .95)],
            "p99_us": s[int(len(s) * .99)], "p999_us": s[min(len(s) - 1, int(len(s) * .999))],
            "max_us": s[-1], "over_1000_us": sum(v > 1000 for v in s)}


def latency_cases(n=1500, seed=40812):
    rng = random.Random(seed)
    observations = []
    for i in range(n):
        street = i % 4
        length = (0, 3, 4, 5)[street]
        c = tuple(rng.sample(range(52), length + 2))
        stacks = [rng.randrange(100, 8000) for j in range(8)]
        own = rng.randrange(0, 150)
        bet = rng.randrange(own, 500)
        bets = [own] + [bet] * 7
        players = tuple(Player(j, stacks[j], bets[j], bets[j] + 200,
                               folded=j > 1 and (i % 3 == 0)) for j in range(8))
        call = min(stacks[0], bet - own)
        history = [Event(3, 0, "raise", 60), Event(0, 0, "call", 60)]
        if street:
            history += [Event(1, street, "raise", bet)]
        if i % 5 == 0:
            history.insert(1, Event(2, 0, "raise", 180))
        payouts = ((50, 30, 20, 0, 0, 0, 0, 0) if i % 2 else None)
        observations.append(Observation(c[:2], c[2:], 0, i % 8, players,
                                        sum(p.invested for p in players), 20, call,
                                        bet + max(20, bet - own), own + stacks[0],
                                        True, tuple(history), i, "latency", payouts))
    return observations


def latency(repetitions):
    cases = latency_cases()
    agent = Agent()
    for obs in cases[:50]:
        agent.act(obs)
    times = []
    cpu_times = []
    streets = [[] for _ in range(4)]
    for i in range(repetitions):
        obs = cases[i % len(cases)]
        t = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        act = agent.act(obs)
        cpu_elapsed = (time.thread_time_ns() - cpu_start) / 1000
        elapsed = (time.perf_counter_ns() - t) / 1000
        Table.validate_action(obs, act)
        times.append(elapsed)
        cpu_times.append(cpu_elapsed)
        streets[obs.street].append(elapsed)
    encoded = [json.dumps(asdict(obs)) for obs in cases[:400]]
    end_to_end = []
    for line in encoded:
        t = time.perf_counter_ns()
        obs = Observation.from_dict(json.loads(line))
        result = json.dumps(agent.act(obs).as_dict())
        end_to_end.append((time.perf_counter_ns() - t) / 1000)
    return {"typed_decision": quantiles(times),
            "thread_cpu_decision": quantiles(cpu_times),
            "wall_over_1ms_with_cpu_under_1ms": sum(w > 1000 and c < 1000 for w, c in zip(times, cpu_times)),
            "by_street": {str(i): quantiles(v) for i, v in enumerate(streets)},
            "json_parse_decide_encode": quantiles(end_to_end),
            "gc_enabled": gc.isenabled(), "case_seed": 40812,
            "note": "Wall time on this host; includes full policy and ICM, excludes import/startup and stdin/stdout IPC. OS preemption may cause outliers."}


def opponent_pool(name):
    if name == "tag":
        return [TightAggressive() for _ in range(7)]
    if name == "loose":
        return [TightAggressive(True) for _ in range(7)]
    if name == "station":
        return [CallingStation() for _ in range(7)]
    if name == "self":
        return [Agent(8921 + j * 1729) for j in range(7)]
    return [TightAggressive(), TightAggressive(True), CallingStation(),
            TightAggressive(), Shove(), CheckFold(), TightAggressive(True)]


def mean_ci(v):
    mean = statistics.mean(v)
    se = statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0
    return {"mean": mean, "standard_error": se, "ci95": [mean - 1.96 * se, mean + 1.96 * se]}


def matches(count, suites, seed_start, structure, deadline_us):
    payouts = {"top3": (50, 30, 20, 0, 0, 0, 0, 0),
               "ordinal": tuple(range(7, -1, -1)),
               "winner": (100, 0, 0, 0, 0, 0, 0, 0)}[structure]
    report = {}
    for suite in suites:
        ours, baseline, differences = [], [], []
        wins = errors = decisions = hands = timeouts = hero_timeouts = 0
        actions = {}
        t0 = time.perf_counter()
        for k in range(count):
            seed = seed_start + k
            result = Table([Agent()] + opponent_pool(suite), seed=seed, payouts=payouts, decision_limit_us=deadline_us).play()
            control = Table([TightAggressive()] + opponent_pool(suite), seed=seed, payouts=payouts, decision_limit_us=deadline_us).play()
            ours.append(result.points[0])
            baseline.append(control.points[0])
            differences.append(result.points[0] - control.points[0])
            wins += result.stacks[0] == 16000
            errors += result.errors + control.errors
            decisions += result.decisions + control.decisions
            hands += result.hands + control.hands
            timeouts += sum(result.timeouts) + sum(control.timeouts)
            hero_timeouts += result.timeouts[0]
            for a, n in result.actions.items():
                actions[a] = actions.get(a, 0) + n
        report[suite] = {"bot_points": mean_ci(ours), "control_tag_points": mean_ci(baseline),
                         "paired_difference": mean_ci(differences), "tables_per_arm": count,
                         "outright_wins": wins, "action_errors": errors,
                         "total_decisions_both_arms": decisions, "total_hands_both_arms": hands,
                         "deadline_timeouts_both_arms": timeouts, "hero_deadline_timeouts": hero_timeouts,
                         "table_actions_bot_arm": actions,
                         "wall_seconds": time.perf_counter() - t0}
        print(json.dumps({suite: report[suite]}), flush=True)
    return {"payouts": payouts, "example_big_blinds": [20, 30, 50, 80, 120, 200, 320, 500],
            "starting_stack": 2000, "hand_cap": 200, "seed_start": seed_start,
            "decision_limit_us": deadline_us,
            "suites": report, "note": "Locally authored simulator and simple opponents; experimental payouts/blind increments, not official data. CI is normal approximation over independent paired seeds; not contest-strength evidence."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("latency", "matches"))
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--tables", type=int, default=100)
    parser.add_argument("--suites", default="tag,loose,station,mixed")
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--payout", choices=("top3", "ordinal", "winner"), default="top3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline-us", type=float, default=1000)
    args = parser.parse_args()
    report = {"python": sys.version, "platform": platform.platform(),
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    inputs = list(Path("pokerbot").glob("*.py")) + [Path("assets/preflop.json"), Path("assets/equity_model.json"),
                                                        Path("tools/simulator.py"), Path("tools/baselines.py"), Path("tools/benchmark.py")]
    report["gameplay_sha256"] = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    if args.mode == "latency":
        report["latency"] = latency(args.iterations)
    else:
        report["matches"] = matches(args.tables, args.suites.split(","), args.seed_start, args.payout, args.deadline_us)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
