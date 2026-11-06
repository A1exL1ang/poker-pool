"""Optional JSON-lines LOCAL adapter (I/O is outside measured Agent.act time).

Input: one observation dict per line, or {"reset": true} between tables.
Output: action dict per observation, {"reset": true} for a reset request.
No official protocol compatibility is claimed.
"""
import json
import sys
from agent import Agent


def main():
    agent = Agent()
    for line in sys.stdin:
        obj = json.loads(line)
        if obj.get('reset') is True:
            agent.reset()
            result = {'reset': True}
        else:
            result = agent.act(obj)
        print(json.dumps(result, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
