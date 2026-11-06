"""Local JSON-lines runner. Process/serialization time is outside act timing."""
import json
import sys
from agent import Agent


def main():
    bot = Agent()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get('command') == 'reset':
                bot.reset(request.get('seed'))
                reply = {'ok': True}
            else:
                reply = bot.act(request.get('observation', request))
        except (ValueError, TypeError, AttributeError):
            reply = {'action': 'fold'}
        print(json.dumps(reply, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
