"""Local integration entry point. This is NOT an official contest adapter.

In-process: from agent import Agent; bot = Agent(); bot.act(observation).
CLI: python agent.py < observations.jsonl
CLI parsing/I/O/startup is separate from the measured in-process decision path.
"""
import json
import sys

from bot.agent import Agent, Config

__all__ = ['Agent', 'Config']


def main():
    agent = Agent()
    for line in sys.stdin:
        obj = json.loads(line)
        if obj.get('reset') is True:
            agent.reset()
            print(json.dumps({'reset': True}), flush=True)
        else:
            print(json.dumps(agent.act(obj), separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
