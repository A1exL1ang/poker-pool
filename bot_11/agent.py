"""Local entry point. This is NOT a claim of official Agent compatibility.

Import Agent, construct it before timing decisions, call .act(observation).
For local JSON-lines smoke tests: python agent.py < observations.jsonl
"""
import json
import sys
from pokerbot.agent import Agent


def main():
    bot = Agent()
    for line in sys.stdin:
        try:
            observation = json.loads(line)
        except (ValueError, TypeError):
            print(json.dumps({'action': 'fold'}), flush=True)
            continue
        print(json.dumps(bot.act(observation), separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
