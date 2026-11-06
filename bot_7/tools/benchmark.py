"""Reproducible synthetic tournament benchmark; no official compatibility claim."""
import argparse
import json
from pathlib import Path
import statistics
import time
from bot7.agent import Agent
from bot7.icm import utility
from .baselines import Baseline
from .simulator import Table
from .evidence import fingerprint


def summarize(values):
    values = sorted(values)
    def q(p):
        return values[min(len(values) - 1, int(p * (len(values) - 1)))] if values else 0
    return {"count": len(values), "mean": statistics.mean(values) if values else 0,
            "median": q(.5), "p95": q(.95), "p99": q(.99), "p999": q(.999), "max": max(values, default=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=int, default=100)
    parser.add_argument("--start-table", type=int, default=0)
    parser.add_argument("--hands", type=int, default=200)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--field", choices=("mixed", "abc", "equity", "station", "nit", "random", "self"), default="mixed")
    parser.add_argument("--blinds", default="20,30,50,80,120,200,320,500", help="Explicit SYNTHETIC blind sequence; changes every 25 hands")
    parser.add_argument("--payouts", default="", help="Optional hypothetical point schedule supplied to the agent")
    parser.add_argument("--comparisons", type=int, default=48)
    parser.add_argument("--budget-us", type=int, default=450, help="0 disables the soft deadline for deterministic strategic experiments")
    parser.add_argument("--enforce-timeout", action="store_true")
    parser.add_argument("--output", default="benchmarks/tournaments.json")
    args = parser.parse_args()
    runtime_fingerprint = fingerprint()
    blinds = [int(x) for x in args.blinds.split(",")]
    payouts = [float(x) for x in args.payouts.split(",")] if args.payouts else ()
    if payouts and len(payouts) != 8:
        parser.error("Supply all eight hypothetical finishing-point values")
    values, points, latencies = [], [], []
    errors = timeouts = wins = hands_played = 0
    runs = []
    start = time.perf_counter()
    for t in range(args.start_table, args.start_table + args.tables):
        hero = t % 8
        seed = args.seed + t * 7919
        agents = []
        for seat in range(8):
            if seat == hero or args.field == "self":
                agents.append(Agent(seed=seed + seat * 13, comparisons=args.comparisons, budget_us=args.budget_us))
            else:
                style = ["abc", "equity", "station", "nit", "random", "abc", "equity", "nit"][seat] if args.field == "mixed" else args.field
                agents.append(Baseline(style, seed + seat * 13))
        table = Table(agents, seed=seed, payouts=payouts, table_id=f"bench-{t}", enforce_timeout=args.enforce_timeout)
        eliminated_points = {}
        for hand in range(args.hands):
            before = table.stacks.copy()
            alive = sum(x > 0 for x in before)
            if alive <= 1:
                break
            table.play_hand(blinds[min(len(blinds) - 1, hand // 25)])
            hands_played += 1
            dead = sorted((i for i in range(8) if before[i] > 0 and table.stacks[i] == 0), key=lambda i: before[i])
            position = alive
            while dead:
                group = [i for i in dead if before[i] == before[dead[0]]]
                # Hypothetical points 8,7,...,1; tied eliminations share places.
                score = statistics.mean(9 - p for p in range(position - len(group) + 1, position + 1))
                for i in group:
                    eliminated_points[i] = score
                    dead.remove(i)
                position -= len(group)
        share = table.stacks[hero] / table.initial_total
        score = eliminated_points.get(hero)
        if score is None:
            score = utility(table.stacks, hero, list(range(8, 0, -1)))
        values.append(share)
        points.append(score)
        wins += share == 1.0
        errors += agents[hero].errors
        timeouts += table.timeouts[hero]
        latencies.extend(table.timings[hero])
        runs.append({"table": t, "seed": seed, "hero_seat": hero, "win_equity": share, "linear_points": score,
                     "final_stacks": table.stacks, "hero_errors": agents[hero].errors})
        if (t + 1) % 25 == 0:
            print(f"{t-args.start_table+1}/{args.tables}: mean win equity={statistics.mean(values):.4f}, linear points={statistics.mean(points):.3f}", flush=True)
    report = {"label": "LOCAL SYNTHETIC BENCHMARK; not official opponents, blind progression, or payouts",
              "configuration": vars(args), "runtime_fingerprint": runtime_fingerprint, "elapsed_seconds": time.perf_counter() - start,
              "hands_played": hands_played, "completed_outright_wins": wins,
              "mean_win_equity": statistics.mean(values),
              "win_equity_standard_error": statistics.stdev(values) / len(values) ** .5 if len(values) > 1 else None,
              "mean_hypothetical_linear_points": statistics.mean(points),
              "linear_points_standard_error": statistics.stdev(points) / len(points) ** .5 if len(points) > 1 else None,
              "symmetric_reference_win_equity": .125, "symmetric_reference_linear_points": 4.5,
              "hero_decision_ns": summarize(latencies), "decisions_over_1ms": timeouts, "agent_errors": errors, "runs": runs}
    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()

