"""Local JSON-lines runner; this is NOT an official contest adapter.

Input one observation object per line. Optional {"reset": true, "seed": 9}
starts a new table. JSON parsing/printing and Agent construction are outside the
strategic act() timing contract. No external services or network calls.
"""
import json
import sys
from bot import Agent


def main():
    agent = Agent()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            observation = json.loads(line)
            if observation.get('reset'):
                agent = Agent(seed=int(observation.get('seed', 9)))
                result = {'reset': True}
            else:
                result = agent.act(observation)
        except (ValueError, TypeError, AttributeError):
            result = {'error': 'Invalid local JSON observation'}
        print(json.dumps(result, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
