# bot_15 local gameplay package

This is one independent offline Python Texas Hold’em bot. It is **not an official-interface-compatible contest submission**. The official starter kit was unavailable. Use the explicit local observation/action contract in `INTERFACE.md` to implement the real adapter and remeasure timing in the actual sandbox.

Python 3.10+; standard library only. Extract this archive, then:

```python
from agent import Agent
bot = Agent(seed=15)        # construction loads/validates assets outside act timing
action = bot.act(observation)
bot.reset()                # required at the beginning of every table
```

Raises are **total street commitments**, calls are incremental chips capped by stack. Pot includes live bets; all-in players stay in `in_hand`. Card IDs use rank `card//4`, suit `card%4`, ranks 2 through Ace. Supply actual finishing-point payouts when known. Omission means chip expected value. Never substitute an experimental payout fixture for the contest's real points.

`python agent.py` is a JSON-lines smoke utility, with `{"reset":true}` to reset. Process startup, JSON parsing, and I/O are outside the in-process decision benchmark. No network, external service, or inference library is used in gameplay. Model assets are numeric JSON, not executable serialized objects.

`MEASUREMENTS.md` reports measured performance, latency overruns, and limitations. `MANIFEST.json` contains SHA-256 hashes for every packaged source/asset/document. Training sources, simulator, tests and complete experiment records are retained in the parent deliverable workspace, not required to run this package.

The policy and learned equity estimates are approximations; no optimality or hard real-time guarantee is claimed. Public contest rules require a 1 ms action; the collector must verify this on its own hardware and actual adapter. Missing assets fail at construction. Internal malformed-input fallbacks are counted in `bot.errors` and choose check when free, otherwise fold.
