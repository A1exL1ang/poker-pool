"""Public local entry point. Adapt this boundary to the official interface."""
import json
import sys
from bot7.agent import Agent
from bot7.interface import Action, Event, Observation, Player


def main():
    agent = Agent()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            print(json.dumps(agent.act_dict(json.loads(line))), flush=True)
        except (ValueError, TypeError, KeyError) as error:
            # Protocol errors are visible; the development runner never silently
            # claims that a malformed observation is an official legal action.
            print(json.dumps({"error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
