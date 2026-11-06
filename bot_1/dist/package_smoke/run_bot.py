"""Persistent JSON-lines harness. JSON parsing/transport is outside act timing."""
import json
import sys
from agent import Agent


def main():
    bot = Agent()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("type") == "reset":
                bot.reset(request.get("table_id"))
                response = {"ok": True}
            elif request.get("type") == "observe":
                bot.observe(request.get("events", []))
                response = {"ok": True}
            else:
                response = bot.act(request.get("observation", request))
        except (ValueError, TypeError, KeyError):
            response = {"action": "fold"}
        print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
