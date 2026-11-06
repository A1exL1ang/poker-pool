"""Optional local JSON-lines adapter; model initialization is outside act()."""
import json
import sys
from agent import Agent


def main():
    bot = Agent()
    for line in sys.stdin:
        try:
            observation = json.loads(line)
            if not isinstance(observation, dict):
                raise ValueError('observation must be an object')
            if observation.get("command") == "reset":
                bot.reset(observation.get("seed", 120012))
                result = {"ok": True}
            else:
                result = bot.act(observation)
        except (ValueError, TypeError):
            result = {"action": "fold"}
        print(json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
