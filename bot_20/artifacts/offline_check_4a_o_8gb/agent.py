"""Local entry point. This is NOT an official contest Agent adapter.

    from agent import Agent
    bot = Agent(seed=20)       # once per table, outside the timed interval
    action = bot.act(observation)

See pokerbot/interface.py and README.md for the exact local contract.
"""

from pokerbot.policy import Agent, PolicyConfig

__all__ = ["Agent", "PolicyConfig"]
