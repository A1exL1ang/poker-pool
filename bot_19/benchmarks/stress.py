"""Synthetic decision stress cases for a bounded remote Linux validation job.

Run on authorized compute, not on the laptop while the user is sleeping:
    python benchmarks/stress.py --iterations 1000 --output measurements/stress.json

Each input is produced by the local engine from a valid deck. Hero is dealt
aces so the opening-range fast path cannot accidentally skip the shove model.
Imports, table reset and observation construction are outside decision timing.
No performance guarantee is inferred from this finite local-hardware sample.
"""

import argparse
import gc
import json
from pathlib import Path
import platform
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import Agent
from benchmarks.run import source_hashes
from equity_model import asset_status
from sim import HandState


def premium_hand(stacks, button, seed, hero, payouts=None):
    """Produce a legitimate deck permutation with Ac/Ad at hero's deal slots."""
    initial = HandState(stacks, button=button, seed=seed, payouts=payouts)
    slots = [initial.deck.index(card) for card in initial.holes[hero]]
    deck = initial.deck[:]
    for slot, card in zip(slots, (48, 49)):
        source = deck.index(card)
        deck[slot], deck[source] = deck[source], deck[slot]
    return HandState(stacks, button=button, seed=seed, deck=deck, payouts=payouts)


def passive(hand):
    hand.apply({"action": "call" if hand.observation()["to_call"] else "check"}, strict=True)


def make_cases():
    cases = {}
    hand = premium_hand([200, 9, 37, 61, 83, 117, 157, 243], 5, 771, 0)
    cases["eight_unequal_unopened"] = hand.observation()

    hand = premium_hand([41, 61, 83, 107, 151, 197, 239, 251], 5, 772, 7)
    while hand.actor != 7:
        passive(hand)
    cases["eight_unequal_limped_bb"] = hand.observation()

    hand = premium_hand([200] * 8, 5, 773, 0)
    cases["eight_equal_unopened"] = hand.observation()

    hand = premium_hand([200, 5, 31, 61, 83, 117, 157, 243], 0, 774, 3)
    cases["eight_with_forced_allin_small_blind"] = hand.observation()

    for name, payouts in (
        ("top3", [.5, .3, .2, 0., 0., 0., 0., 0.]),
        ("linear", [8 / 36, 7 / 36, 6 / 36, 5 / 36, 4 / 36, 3 / 36, 2 / 36, 1 / 36]),
        ("satellite", [1., 1., 0., 0., 0., 0., 0., 0.]),
    ):
        hand = premium_hand([500] * 8, 0, 775, 2, payouts=payouts)
        hand.apply({"action": "raise", "amount": 500}, strict=True)
        while hand.actor != 2:
            hand.apply({"action": "fold"}, strict=True)
        cases["eight_live_seats_isolated_icm_" + name] = hand.observation()
    return cases


def summarize(samples):
    samples = sorted(samples)
    return {
        "count": len(samples), "mean_us": statistics.mean(samples) / 1000,
        **{f"p{p}_us": samples[min(len(samples) - 1, int(len(samples) * p / 100))] / 1000 for p in (50, 90, 99, 99.9)},
        "max_us": samples[-1] / 1000,
        "over_1ms": sum(value > 1_000_000 for value in samples),
    }


def legal(obs, action):
    if action.get("action") not in obs["legal_actions"]:
        return False
    if action["action"] != "raise":
        return True
    amount = action.get("amount")
    return (
        type(amount) is int and obs["bets"][obs["hero"]] + obs["to_call"] < amount <= obs["max_raise_to"]
        and (amount >= obs["min_raise_to"] or amount == obs["max_raise_to"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--params")
    parser.add_argument("--output", default="measurements/stress.json")
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100000:
        parser.error("iterations must be between1 and100000")
    params = json.loads(Path(args.params).read_text()) if args.params else None
    report = {
        "kind": "synthetic valid-state stress; local assumptions, no official interface claim",
        "arguments": vars(args), "hashes": source_hashes(), "assets": asset_status(),
        "python": platform.python_version(), "platform": platform.platform(),
        "clock_boundary": "in-process act only; imports, reset and observation construction excluded",
        "thread_clock": vars(time.get_clock_info("thread_time")), "gc_enabled": gc.isenabled(),
        "cases": {},
    }
    for name, obs in make_cases().items():
        bot = Agent(params=params)
        wall, cpu = [], []
        overruns = []
        invalids = fallbacks = 0
        batch_cpu = time.thread_time_ns()
        batch_wall = time.perf_counter_ns()
        for index in range(args.iterations):
            bot.reset()  # outside each measured decision
            start_cpu = time.thread_time_ns()
            start_wall = time.perf_counter_ns()
            action = bot.act(obs)
            elapsed = time.perf_counter_ns() - start_wall
            used = time.thread_time_ns() - start_cpu
            wall.append(elapsed)
            cpu.append(used)
            invalids += not legal(obs, action)
            fallbacks += bot.fallbacks
            if elapsed > 1_000_000:
                overruns.append({"index": index, "wall_us": elapsed / 1000, "raw_thread_cpu_us": used / 1000})
        batch_wall = time.perf_counter_ns() - batch_wall
        batch_cpu = time.thread_time_ns() - batch_cpu
        zeros = sum(value == 0 for value in cpu)
        report["cases"][name] = {
            "hero": obs["hero"], "hole": obs["hole"], "history_events": len(obs["history"]),
            "observation": obs, "wall": summarize(wall), "thread_cpu_raw": summarize(cpu),
            "first_wall_us": wall[0] / 1000, "first_thread_cpu_raw_us": cpu[0] / 1000,
            "zero_thread_cpu_samples": zeros,
            "thread_cpu_interpretation": "quantized; use aggregate batch CPU, not individual samples" if zeros > len(cpu) * .1 else "individual samples available; OS and clock semantics still apply",
            "batch_wall_us_per_iteration_including_reset_and_logging": batch_wall / args.iterations / 1000,
            "batch_thread_cpu_us_per_iteration_including_reset_and_logging": batch_cpu / args.iterations / 1000,
            "invalid_actions": invalids, "internal_fallbacks": fallbacks, "overruns": overruns,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in result.items() if key not in ("observation", "overruns")} for name, result in report["cases"].items()}, indent=2))
    if any(result["invalid_actions"] or result["internal_fallbacks"] for result in report["cases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
