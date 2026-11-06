"""Diagnostic JSON-lines runner for the documented LOCAL interface.

One observation per line, one action per line. {"reset": true} resets the table.
Serialization/process overhead is outside Agent.act's in-process timing report.
"""
import json
import sys
from agent import Agent


def main():
    agent=Agent()
    for line in sys.stdin:
        try:
            observation=json.loads(line)
            if isinstance(observation,dict) and observation.get('reset') is True:
                agent.reset(); action={'reset':True}
            else:
                action=agent.act(observation)
        except (ValueError,TypeError):
            action={'action':'fold'}
        sys.stdout.write(json.dumps(action,separators=(',',':'))+'\n')
        sys.stdout.flush()


if __name__=='__main__': main()
