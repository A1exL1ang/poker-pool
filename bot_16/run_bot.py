"""Optional offline JSON-lines runner for the LOCAL contract, not official I/O.

Each input line is an observation. {"reset": true, "seed": 16} resets the table.
The 1ms policy benchmark excludes JSON/process transport; official transport is
unknown. A persistent process avoids paying import/model initialization per turn.
"""
import argparse
import json
import sys

from agent import Agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=16)
    args = parser.parse_args()
    agent = Agent(seed=args.seed)
    for line in sys.stdin:
        try:
            observation = json.loads(line)
            if observation.get('reset') is True:
                agent.reset(observation.get('seed', args.seed))
                response = {'reset': True}
            else:
                response = agent.act(observation)
        except (ValueError, TypeError, AttributeError):
            response = {'error': 'invalid local-contract JSON observation'}
        print(json.dumps(response, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
