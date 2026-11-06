"""Local JSONL adapter: one observation in, one action out. Never networks."""

import json
import sys
from pokerbot import Agent, Observation


def main():
    agent = Agent()
    for line in sys.stdin:
        if not line.strip():
            continue
        raw = None
        try:
            raw = json.loads(line)
            obs = Observation.from_dict(raw)
            action = agent.act(obs)
            print(json.dumps(action.as_dict(), separators=(",", ":")), flush=True)
        except Exception as exc:
            # Fail visibly at the integration boundary; the strategy has its own
            # legal-action gate. Diagnostics never contaminate stdout.
            print(json.dumps({"error": str(exc)}), file=sys.stderr, flush=True)
            fallback = "check" if isinstance(raw, dict) and raw.get("to_call") == 0 and "check" in raw.get("legal", ("check",)) else "fold"
            print(json.dumps({"action": fallback, "amount": 0}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
