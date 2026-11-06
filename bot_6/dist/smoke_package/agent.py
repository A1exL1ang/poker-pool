"""Local import adapter and persistent JSON-lines runner.

Import Agent and call act(observation), or: python agent.py < observations.jsonl
Parsing/startup are separate from the strategy timing in our benchmarks.
"""
import json
import sys
from bot6.agent import Agent


def main():
    agent = Agent()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            observation = json.loads(line)
            action = agent.act(observation)
            print(json.dumps(action, separators=(",", ":")), flush=True)
        except Exception as exc:
            # Errors are explicit on stderr, not silently passed as valid
            # official observations. A caller may apply its own fallback.
            print(type(exc).__name__ + ": " + str(exc), file=sys.stderr, flush=True)
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
