"""Submission-facing import and optional JSON-lines adapter for OUR local schema.

Official contest interface was not supplied; adaptation will be required.
"""
from pokerbot.agent import Agent


def main():
    import json
    import sys
    bot = Agent()
    for line in sys.stdin:
        try:
            obs = json.loads(line)
            action = bot.act(obs)
        except (ValueError, TypeError):
            action = {"action": "fold", "amount": 0}
        sys.stdout.write(json.dumps(action, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
