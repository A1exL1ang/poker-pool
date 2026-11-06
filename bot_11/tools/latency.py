"""Measure full offline Agent.act latency without discarding scheduling outliers.

Run from the workspace root: python -m tools.latency --samples 10000 --enforce
Initial asset loading and Agent construction are measured separately. Timed
decisions include observation validation, opponent statistics, feature/model
inference, strategy, ICM when used, and legal-action normalization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
from time import perf_counter, perf_counter_ns

import pokerbot.agent as policy
from pokerbot.baselines import make_baseline
from pokerbot.engine import Tournament


STYLES = ("tag", "lag", "station", "tight_passive", "pushfold", "tag", "random")
SYNTHETIC_PAYOUTS = [.5, .3, .2, 0., 0., 0., 0., 0.]
LIMIT_NS = 1_000_000


def quantiles(durations):
    """Nearest-rank percentiles over every measured decision."""
    if not durations:
        return {"samples": 0}
    ordered = sorted(durations)
    def at(p):
        return ordered[max(0, math.ceil(len(ordered) * p) - 1)] / 1000
    over = sum(t > LIMIT_NS for t in durations)
    return {"samples": len(durations), "mean_us": sum(durations) / len(durations) / 1000,
            "p50_us": at(.5), "p90_us": at(.9), "p99_us": at(.99),
            "p99_9_us": at(.999), "max_us": ordered[-1] / 1000,
            "over_1ms": over, "over_1ms_fraction": over / len(durations)}


def legal_action(obs, action):
    if not isinstance(action, dict) or action.get("action") not in obs["legal_actions"]:
        return False
    if action["action"] != "raise":
        return True
    amount = action.get("amount")
    return (isinstance(amount, int) and not isinstance(amount, bool) and obs["can_raise"] and
            obs["current_bet"] < amount <= obs["max_raise_to"] and
            (amount >= obs["min_raise_to"] or amount == obs["max_raise_to"]))


class Collector:
    def __init__(self, agent, records, table, limit):
        self.agent, self.records, self.table, self.limit = agent, records, table, limit

    def __call__(self, obs):
        # The engine creates fresh observation dictionaries with copied public
        # state. Keep the pre-decision object; the policy does not mutate it.
        if len(self.records) < self.limit:
            self.records.append((self.table, obs))
        return self.agent.act(obs)


def capture(samples, seed, max_tables, enforce):
    records, tables = [], []
    totals = Counter()
    for table in range(max_tables):
        if len(records) >= samples:
            break
        table_seed = seed + table
        hero = table_seed % 8
        candidate = policy.Agent(seed=table_seed ^ 110011)
        wrapper = Collector(candidate, records, table, samples)
        agents, other = [], 0
        for seat in range(8):
            if seat == hero:
                agents.append(wrapper)
            else:
                agents.append(make_baseline(STYLES[(other + table) % 7], table_seed * 17 + seat))
                other += 1
        # These are transparent local hypotheses; official payouts are absent.
        payouts = SYNTHETIC_PAYOUTS if table % 4 == 3 else None
        result = Tournament(agents, seed=table_seed, payouts=payouts,
                            enforce_time_limit=enforce).run()
        for key in ("hands", "decisions", "errors", "timeouts", "invalid_actions"):
            totals[key] += result[key]
        totals["policy_error_fallbacks"] += candidate.diagnostics["errors"]
        tables.append({"seed": table_seed, "hero_seat": hero, "hands": result["hands"],
                       "policy_decisions": candidate.diagnostics["decisions"],
                       "synthetic_payouts": payouts is not None})
    return records, {"tables": len(tables), "totals": dict(totals), "table_seeds": tables}


def stress_observations(count, seed):
    """Eight alive seats, six folded, and a large HU call that invokes ICM.

    Distinct random cards cover all streets. Large-stack disparity and all-in
    villain exercise the exact subset DP. These are synthetic valid public
    states, separate from naturally captured tournament observations.
    """
    rng = random.Random(seed)
    output = []
    for i in range(count):
        board_size = (0, 3, 4, 5)[i % 4]
        cards = rng.sample(range(52), board_size + 2)
        street = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[board_size]
        mine = rng.randrange(300, 2500)
        invested = rng.randrange(20, 150)
        stacks = [mine, 0] + [rng.randrange(1, 12000) for _ in range(6)]
        committed = [invested, invested + mine] + [0] * 6
        bets = [invested, invested + mine] + [0] * 6 if not board_size else [0, mine] + [0] * 6
        current = max(bets)
        players = [{"seat": seat, "stack": stacks[seat], "committed": committed[seat],
                    "street_bet": bets[seat], "folded": seat >= 2,
                    "all_in": seat == 1} for seat in range(8)]
        history = [{"seat": 1, "street": "preflop", "action": "raise",
                    "amount": current if not board_size else invested}]
        if board_size:
            history.extend([{"seat": 0, "street": "preflop", "action": "call", "amount": invested},
                            {"seat": 1, "street": street, "action": "raise", "amount": mine}])
        obs = {"hole_cards": cards[:2], "board": cards[2:], "seat": 0, "button": 7,
               "small_blind": 10, "big_blind": 20, "small_blind_seat": 0,
               "big_blind_seat": 1, "hand_number": i, "street": street,
               "players": players, "pot": sum(committed), "to_call": mine,
               "current_bet": current, "min_raise_to": current * 2,
               "max_raise_to": bets[0] + mine, "can_raise": False,
               "legal_actions": ["fold", "call"], "history": history,
               "payouts": SYNTHETIC_PAYOUTS[:]}
        output.append((i, obs))
    return output


def timed_call(agent, obs):
    previous_errors = agent.diagnostics["errors"]
    caught = None
    start = perf_counter_ns()
    try:
        action = agent.act(obs)
    except Exception as exc:
        action = {"action": "check" if "check" in obs["legal_actions"] else "fold"}
        caught = f"{type(exc).__name__}: {exc}"
    elapsed = perf_counter_ns() - start
    return elapsed, action, agent.diagnostics["errors"] - previous_errors, caught


def new_stats():
    return {"durations_ns": [], "policy_error_fallbacks": 0, "uncaught_exceptions": 0,
            "invalid_actions": 0, "actions": Counter(), "worst": []}


def record(stats, index, obs, measurement):
    elapsed, action, errors, caught = measurement
    stats["durations_ns"].append(elapsed)
    stats["policy_error_fallbacks"] += errors
    stats["uncaught_exceptions"] += caught is not None
    stats["invalid_actions"] += not legal_action(obs, action)
    stats["actions"][action.get("action", "invalid")] += 1
    item = {"sample": index, "elapsed_us": elapsed / 1000, "street": obs["street"],
            "hand_number": obs["hand_number"], "board": obs["board"],
            "hole_cards": obs["hole_cards"], "history_events": len(obs["history"]),
            "live_opponents": sum(not p["folded"] and p["seat"] != obs["seat"] for p in obs["players"]),
            "synthetic_or_supplied_payouts": obs["payouts"] is not None,
            "to_call": obs["to_call"], "action": action, "exception": caught}
    stats["worst"].append(item)
    stats["worst"].sort(key=lambda x: x["elapsed_us"], reverse=True)
    del stats["worst"][10:]


def finish_stats(stats):
    stats.update(quantiles(stats["durations_ns"]))
    stats["deadline_fallbacks_if_enforced"] = stats.get("over_1ms", 0)
    stats["actions"] = dict(stats["actions"])
    return stats


def replay(records, seed):
    cold, warm, sequential = new_stats(), new_stats(), new_stats()
    init_times = []
    for index, (_, obs) in enumerate(records):
        start = perf_counter_ns()
        agent = policy.Agent(seed=seed + index)
        init_times.append(perf_counter_ns() - start)
        record(cold, index, obs, timed_call(agent, obs))
        record(warm, index, obs, timed_call(agent, obs))
    agent = None
    table = None
    for index, (table_id, obs) in enumerate(records):
        if table_id != table:
            table = table_id
            agent = policy.Agent(seed=seed + table_id)
        record(sequential, index, obs, timed_call(agent, obs))
    return {"fresh_state_cold_board": finish_stats(cold),
            "immediate_repeat_warm_board": finish_stats(warm),
            "sequential_table_replay": finish_stats(sequential),
            "agent_construction_after_assets": quantiles(init_times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--stress-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1201100)
    parser.add_argument("--max-tables", type=int, default=1000)
    parser.add_argument("--enforce", action="store_true", help="Apply local 1ms fallback during tournament capture")
    parser.add_argument("--out", default="benchmarks/latency.json")
    args = parser.parse_args()
    if args.samples < 1 or args.stress_samples < 0 or args.max_tables < 1:
        parser.error("samples/max-tables must be positive; stress-samples nonnegative")
    wall = perf_counter()
    start = perf_counter_ns()
    policy.load_assets()
    asset_ns = perf_counter_ns() - start
    paths = [Path("pokerbot/agent.py"), Path("pokerbot/features.py"), Path("pokerbot/icm.py"),
             Path("pokerbot/equity.py"), Path("assets/equity.bin.z"), Path("assets/preflop.json")]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    records, capture_report = capture(args.samples, args.seed, args.max_tables, args.enforce)
    report = {"arguments": vars(args), "platform": platform.platform(),
              "python": sys.version, "clock": "perf_counter_ns wall time; no outliers removed",
              "assumptions": "Independent local NLHE simulator; assumed doubling blind schedule; any payouts are explicitly synthetic, never official. No hard-real-time guarantee on other hardware.",
              "timing_scope": "Full Agent.act only. Asset load and Agent construction reported separately. Cold uses a fresh Agent per observation; warm immediately repeats it. Sequential reconstructs table order with fresh Agents at table boundaries.",
              "asset_load_us": asset_ns / 1000, "requested_observations": args.samples,
              "captured_observations": len(records),
              "street_counts": dict(Counter(obs["street"] for _, obs in records)),
              "capture": capture_report, "realistic": replay(records, args.seed ^ 515151)}
    stress = stress_observations(args.stress_samples, args.seed ^ 717171)
    if stress:
        report["synthetic_8seat_icm_stress"] = replay(stress, args.seed ^ 818181)
    report["sources"] = source_hashes
    report["source_changed_during_run"] = [str(path) for path in paths
                                           if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[str(path)]]
    report["seconds"] = perf_counter() - wall
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    concise = {"output": str(output), "captured_observations": len(records),
               "asset_load_us": report["asset_load_us"], "capture_totals": capture_report["totals"],
               "seconds": report["seconds"]}
    for group in ("realistic", "synthetic_8seat_icm_stress"):
        if group in report:
            concise[group] = {mode: {key: value for key, value in data.items()
                                    if key not in ("durations_ns", "worst")}
                              for mode, data in report[group].items()}
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
