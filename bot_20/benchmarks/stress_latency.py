"""Legal adversarially long betting histories; run on a private compute job."""
import json
from pathlib import Path
from time import perf_counter_ns

from agent import Agent
from pokerbot.interface import call, is_legal, raise_to
from simulator.engine import Hand
from benchmarks.latency import percentiles


def main():
    corpus = []
    lengths = []
    for j, stacks in enumerate(([2000] * 8, [6000, 6000, 4000, 0, 0, 0, 0, 0], [8000, 8000, 0, 0, 0, 0, 0, 0])):
        hand = Hand(stacks, button=0, seed=3000 + j, table_id=3000 + j,
                    payouts=[.5, .3, .2, 0, 0, 0, 0, 0])
        observations = []
        while not hand.done:
            obs = hand.observation()
            observations.append(obs)
            pending = [i for i in range(8) if not hand.folded[i] and hand.stacks[i] > 0
                       and (hand.last_acted_at[i] is None or hand.street_committed[i] < hand.current_bet)]
            action = raise_to(obs, obs["min_raise_to"]) if len(pending) == 1 and obs["can_raise"] else call(obs)
            hand.act(action)
        selected = sorted(observations, key=lambda x: len(x["history"]), reverse=True)[:20]
        corpus.extend(selected)
        lengths.append(max(len(x["history"]) for x in observations))
    bot = Agent(seed=320)
    if not bot.estimator.model_loaded or not bot.estimator.preflop_loaded:
        raise RuntimeError("Trained assets required")
    times = []
    illegal = 0
    for repeat in range(30):
        bot.reset()
        for obs in corpus:
            start = perf_counter_ns()
            action = bot.act(obs)
            elapsed = perf_counter_ns() - start
            times.append(elapsed)
            illegal += not is_legal(obs, action)
    result = {"purpose": "Legal long-history timing stress; not playing-strength evidence", "history_maxima": lengths,
              "timing": percentiles(times), "illegal": illegal, "internal_errors": bot.errors,
              "internal_budget_fallbacks": bot.budget_fallbacks}
    Path("measurements").mkdir(exist_ok=True)
    Path("measurements/latency_stress.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
