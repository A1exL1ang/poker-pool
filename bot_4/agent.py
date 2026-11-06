"""Integration entry point. Local contract is documented in README.md."""
from bot.policy import Agent
from bot.interface import State, Player, Event, Action, validate_state, legalize

__all__ = ["Agent", "State", "Player", "Event", "Action", "validate_state", "legalize"]


if __name__ == "__main__":
    import json
    import sys
    bot = Agent()
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(bot.act(json.loads(line)), separators=(",", ":")), flush=True)
