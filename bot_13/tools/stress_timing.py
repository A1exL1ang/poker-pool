"""Synthetic worst-range full-callback timing, distinct from poker evaluation.

Run on an owned compute node. Uses valid public chip/card states but injects
concentrated internal range models to exercise bounded rejection and ICM.
"""

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bot13.agent import Agent
from bot13.interface import validate, legalize
from tools.latency import stats


def observation(n, board):
    players = [{"seat": i, "stack": 1000 if i == 0 else 800,
                "bet": 0 if i == 0 else 200, "committed": 1000 if i == 0 else 1200,
                "folded": False, "all_in": False} for i in range(n)]
    return {"hand_id": 1, "hero": 0, "button": n - 1, "seat_count": n,
            "small_blind_seat": 0, "big_blind_seat": 1, "small_blind": 10, "big_blind": 20,
            "street": {3: "flop", 4: "turn", 5: "river"}[len(board)],
            "hole_cards": [48, 49], "board": board, "pot": sum(p["committed"] for p in players),
            "to_call": 200, "current_bet": 200, "min_raise_to": 400, "max_raise_to": 1000,
            "can_raise": True, "players": players, "history": [],
            "payouts": [50, 30, 20, 0, 0, 0, 0, 0][:n]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cases = []
    for n, board, tight, native in (
        (2, [0, 5, 10], False, True), (2, [0, 5, 10, 15, 20], True, True),
        (8, [0, 5, 10], False, True), (8, [0, 5, 10], True, True),
        (8, [0, 5, 10, 15], True, True), (8, [0, 5, 10, 15, 20], True, True),
        (8, [0, 5, 10, 15], True, False), (6, [0, 5, 10, 15], True, False),
    ):
        obs = observation(n, board)
        validate(obs)
        agent = Agent(seed=133100, config={"native": native})
        agent._update(obs)
        if tight:
            for model in agent.opponents.values():
                model.width, model.threshold, model.slope = 0.03, 0.85, 0.05
                model.bluff, model.draw = 0.05, 0.15
        elapsed = []
        illegal = 0
        for _ in range(a.iterations):
            start = perf_counter_ns()
            action = agent.act(obs)
            elapsed.append(perf_counter_ns() - start)
            illegal += action != legalize(action, obs)
        cases.append({"players": n, "street": obs["street"], "tight_models": tight,
                      "native": agent.native is not None, "timing": stats(elapsed), "illegal": illegal,
                      "fallbacks": agent.fallbacks, "native_errors": agent.native_errors})
    result = {"synthetic_internal_model_stress_not_playing_strength": True,
              "platform": platform.platform(), "cases": cases}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
