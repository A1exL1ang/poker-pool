"""JSON-lines local adapter. Keep one process/Agent alive per table."""
import json
import sys

from .agent import Agent
from .interface import validate_action, validate_observation


def main():
    agent = Agent()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            o = json.loads(line)
            if o.get("command") == "reset":
                agent.reset(o.get("table_id"))
                result = {"ok": True}
            else:
                validate_observation(o)
                result = validate_action(o, agent.act(o))
        except (KeyError, TypeError, ValueError) as e:
            # Report invalid input explicitly; the engine controls timeout fallback.
            result = {"error": str(e)}
        sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
