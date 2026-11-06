"""LOCAL JSON-lines runner. No claim of official host protocol compatibility.

Each input line is a canonical observation, or {"reset": true, "seed": 13}.
Timing of parsing and transport is not included in local act() measurements.
"""

import argparse
import json
import sys
from .agent import Agent
from .interface import safe_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    agent = Agent(seed=args.seed)
    for line in sys.stdin:
        try:
            obs = json.loads(line)
            if isinstance(obs, dict) and obs.get("reset") is True:
                agent.reset(seed=obs.get("seed"))
                result = {"reset": True}
            else:
                result = agent.act(obs)
        except (ValueError, TypeError):
            result = {"action": "fold"}
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
