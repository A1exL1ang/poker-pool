# bot_16 — independent offline no-limit Hold'em agent

This is a local research implementation, **not a contest submission or an
official-interface-compatible bot**. The complete supplied `RULES.txt` was read.
No starter kit, official observation schema, point distribution, or full blind
schedule was available. `CONTRACT.md` defines every local input/action convention;
`SIMULATOR.md` identifies simulator assumptions. An official adapter is still
required when the real starter kit becomes available.

## Run

Python 3.11 or newer, standard library only for gameplay:

```python
from agent import Agent
agent = Agent(seed=16)             # construct once per table
action = agent.act(observation)   # local dictionary contract
agent.reset(seed=17)              # clear all table-local opponent information
```

`python run_bot.py` provides an optional persistent JSON-lines wrapper. It is a
local convenience protocol. Process startup, JSON transport, and the official
sandbox's timing boundary are unknown and are not claimed to fit in 1ms.

Keep `agent.py`, the entire `bot/` directory, and the required `assets/*.json`
files together. Runtime makes no network calls, runs no subprocesses, and needs
no external packages. Model initialization happens during import. Offline
training tools are separate and use C++/OpenMP and pinned NumPy/scikit-learn.
`dist/bot_16.zip` is the compact runtime package; the surrounding workspace
contains the tests, training source, datasets, benchmark commands, and reports.

## Strategy

The policy combines positional preflop ranges, stack-dependent sizing and shove
estimates, original Monte Carlo preflop tables, and a learned postflop equity
model. Model targets account for card removal, ties, multiway pots, several
starting-hand ranges, and explicit synthetic betting-range assumptions. Static
generated Python evaluates the fitted trees; there is no live Monte Carlo,
model API, or search service. The fast hand evaluator and the simulator's
independent reference evaluator are checked against each other.

Observed public actions update anonymous seat statistics with shrinkage toward
priors. All state resets per table. The bot has no knowledge of opponent teams,
private cards, future board cards, or other bots' implementations. Betting
history is incrementally summarized to keep long histories cheap.

The default objective is chip value when payouts are absent. When caller-supplied
payouts and a suitable heads-up all-in situation are available, an exact subset
ICM calculation adjusts the call threshold. Payouts are never invented by the
runtime. This is a heuristic tournament policy, not an equilibrium solver, and
ICM call calculations simplify future skill and split-pot outcomes.

Opponent ranges and bet pressure are empirical assumptions. The model uses a
common representative range for several opponents; layered side-pot equity is
approximated, although action sizing, eligible pot amounts, and simulator
settlement handle side pots explicitly. Rare river estimates can still have
large errors. Exact guaranteed-win and shared-board safeguards cover only
provable cases. These limits are retained in the final evidence rather than
described as solved poker.

## Reproduce

```text
python -m unittest discover -s tests -v
python benchmarks/fuzz.py --hands 10000 --seed 161616
python benchmarks/tournament.py --tables 200 --seed 900000 --field mixed --output benchmarks/example.json
python benchmarks/tune.py --tables 96 --seed 160000 --output benchmarks/tuning.json
```

The benchmark's point schedules and later blind levels are explicitly
**synthetic**. Its default initial 2,000 chips, 10/20 blinds, eight seats,
25-hand level changes, and 200-hand cap come from the supplied rules. Results
against independently authored heuristic opponents do not establish strength
against unseen contest entrants. Tuning and held-out runs use disjoint seeds.
See `training/README.md` for data/model reproduction, seeds, assumptions, and
training-only dependencies. Final measured results and limitations are recorded
in `MEASUREMENTS.md` after the candidate is frozen.

## Design references

The [Pluribus paper](https://doi.org/10.1126/science.aay2400) motivates separating
offline work from inexpensive runtime decisions and evaluating multiplayer
performance empirically. This implementation does not reproduce Pluribus's
training algorithm or claim its strength. Standard raise/reopening behavior is
checked against the [Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/).
All strategy code, evaluators, generated data, and synthetic opponents were
developed independently within this bot's workspace.

No code has been pushed, published, or submitted. Remote experiments use only
new bounded jobs belonging to this bot; their IDs and final states are recorded
in `.control/jobs.json`. No other team's workspace or pre-existing allocation
was inspected or used.

After the user's quiet-laptop instruction, all remaining substantial tuning and
validation runs were moved to new bounded remote allocations. The interrupted
local confirmation run is not used as held-out evidence.
