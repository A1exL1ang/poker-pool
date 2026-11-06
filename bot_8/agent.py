"""Public local entry point. Adapt official observations here when provided.

    from agent import Agent
    action = Agent().act(observation)  # keep one Agent instance per table!

CLI: python agent.py --jsonl < examples/observations.jsonl
The timed gameplay path is Agent.act, not process startup or JSON transport.
"""

import argparse
import json
import sys

from kestrel.model import Action, Event, Observation, Player
from kestrel.policy import Agent

__all__ = ["Agent", "Action", "Observation", "Player", "Event"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="store_true", help="read one local observation per line")
    parser.parse_args()
    bot = Agent()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            action = bot.act(json.loads(line))
            print(json.dumps(action.as_dict(), separators=(",", ":")), flush=True)
        except (ValueError, TypeError, KeyError) as error:
            # Protocol errors are explicit; never silently invent an observation.
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
