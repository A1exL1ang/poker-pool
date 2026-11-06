# bot_19

An independent, offline Python bot for standard no-limit Texas Hold'em. This is
a local experimental implementation, not a contest submission. The official
starter kit and Agent interface were not supplied. See [INTERFACE.md](INTERFACE.md)
before adapting it to any external runner.

```python
from agent import Agent
bot = Agent(seed=19)       # create once per anonymous table
action = bot.act(observation)
```

The delivered runtime uses only Python's standard library. Training dependencies
are unnecessary for gameplay. It reads its own assets during module import and
does no file, network, outside API, or external model-service calls during decisions. Use a fresh
instance, or `reset()`, at each new table. No cross-table opponent identities or
state are stored.

Measured results, held-out comparisons, timing overruns, and final validation
counts are recorded in [RESULTS.md](RESULTS.md). The offline runtime archive is
`dist/bot_19.zip`; source, tests, and reproducible tooling remain in this workspace.

## Strategy

- Monte Carlo preflop pot-share tables for all 169 hand classes, six opponent
  range widths, and one through seven opponents, with joint card removal.
- Original hand evaluator, board/draw features, and offline-trained compact tree
  models for postflop equity against a declared synthetic range model: 192 trees
  per street, depth eight, 43 base plus 12 derived features. Generated Python
  inference is bound to the source model by SHA256.
- Separately calibrated multiway equity and tie-aware handling when the board
  plays, with exact checks for several universally unbeatable shared boards.
- Position and stack dependent opening, isolation and re-raising; shallow-stack
  shove estimates with caller-subset side-pot accounting; pot-odds calls with
  equity-realization discounts, including short all-in blinds.
- Public-history range estimates and conservative, prior-shrunk opponent
  frequencies. A player who raises and then calls retains the raising evidence.
- Value and occasional draw/blocker bluffs with a small set of bet sizes.
- Exact ICM call thresholds for supported isolated heads-up all-in situations
  when the observation explicitly supplies payouts. Otherwise chip-equity logic.

This is a trained equity estimator plus a tuned decision policy. It is **not** a
full-game CFR strategy, a proven equilibrium, or a guarantee against arbitrary
opponents. The public Pluribus research motivated offline abstraction and mixed
actions; no Pluribus implementation, opponent bot, or private repository was used.

## Local mechanics and assumptions

The complete `RULES.txt` was read. Known contest constraints are eight seats,
2,000 starting chips, 10/20 initial blinds, increases every 25 hands, a 200-hand
cap, offline execution, a 1 ms decision deadline, and a 100 MB file-size limit.
The user specified standard no-limit Texas Hold'em and explicitly removed the
historical event's one-hour development limit for this task.

The official action schema, exact blind increases, and finishing points remain
unknown. `sim.py` is an original simulator with explicit local schedules and
optional supplied payouts. No payout vector here is represented as official.
The benchmark's `chip`, `top3`, and `linear` cases are sensitivity experiments.
For rule and observation details, see [SIMULATOR.md](SIMULATOR.md).

Raises use **total street commitment**. The pot includes current-street bets.
Calls are capped at the acting stack. Short all-ins, reopening, side pots,
uncalled refunds, heads-up blinds, dead buttons, tied eliminations, and odd chips
are tested. Only the hero's hole cards and public history appear in observations.

## Reproduce validation

Run from this directory with Python 3.11 or newer:

```text
python -m unittest discover -s tests -v
python training/validate_native.py native/equity.exe
python training/export_fast.py
python training/validate_fast.py
python benchmarks/run.py --games 1200 --split validation --workers 4
python benchmarks/latency.py --tables 160 --repeat 3
python benchmarks/stress.py --iterations 1000
```

Native validation requires first compiling the original C++ generator; it is not
a gameplay dependency. Windows: `g++ -O3 -std=c++17 -fopenmp native/equity.cpp -o native/equity.exe`.
Linux uses the same command with output `native/equity`.

`benchmarks/run.py` rotates seats, uses reproducible deck seeds and original local
sparring policies, and records errors, invalid actions, fallback counts, timing,
individual table outcomes, source hashes, and confidence intervals. Run with
`--enforce` to replace late decisions by check/fold. Normal tuning records but
does not enforce wall time, to keep strategy comparisons reproducible under OS
scheduling noise.

`benchmarks/tune.py` uses tune seeds starting at 1,910,000 and independent
validation seeds starting at 1,950,000. The final tournament test range begins at
19,900,000 and must not be used for further parameter selection. Range-equity
training, validation, and test labels use separately recorded seeds and hashes.
The final protocol is `benchmarks/final_evaluate.py`.
Measurement JSON files distinguish development, tuning, validation, and held-out
results; development runs may explicitly use a heuristic missing-asset fallback.
Distribution validation must reject missing trained assets.

## Reproduce training

`training/job.sh` documents the initial bounded CPU training pipeline. It compiles
our native generator, verifies evaluator and feature parity, builds the preflop
table, generates 600,000 training and 45,000 validation states, fits the model,
and evaluates independent labels. It requests 8 CPUs, 32 GB, and 1h50; it is an
example for the explicitly authorized private Engaging directory, not a command
to use another allocation. `training/refine_job.sh` documents the later model
capacity comparison and fresh final equity test.

Remote resources are tracked in `.control/JOBS.json`. No pre-existing user
allocations are used. No gameplay depends on remote resources. Source and data
generation are original; NumPy/XGBoost are offline training tools only.

## Limits and interpretation

The synthetic range model is an assumption about likely holdings after action,
not knowledge of an opponent's actual cards. Approximate range conditioning,
multiway conversion, future equity realization, fold estimates, and bet EV can
all be wrong. Exact ICM is only used in a restricted terminal-call case; side-pot
ICM and future tournament strategy are approximated. Unknown payout/blind rules
could materially change the best policy.

Results against these original stylized baselines do not establish performance
against contest bots or an optimal strategy. Confidence intervals describe table
sampling uncertainty under the stated local experiment, not uncertainty about
the eventual contest environment.

The mirror pool freezes opponent parameter values, while sharing the final
policy code and equity assets. Code/assets must be frozen for all compared policy
candidates. Before/after model-development runs are diagnostics rather than
paired comparisons against unchanged opponents.

Timing measures in-process `Agent.act`. Imports, asset loading, process startup,
JSON IO, and observation construction are outside that boundary. The official
timer boundary and hardware are unknown. Wall-clock overruns are reported,
including those during concurrent workloads; no absolute 1 ms guarantee is
claimed. Python's thread-CPU clock on this Windows environment is quantized and
should not be used to attribute individual stalls.

## Public references

- [Brown & Sandholm, *Superhuman AI for multiplayer poker*](https://noambrown.github.io/papers/19-Science-Superhuman.pdf): offline strategy abstraction and mixed actions. Our policy is substantially simpler and does not implement their solver.
- [Ganzfried & Sandholm, *Safe Opponent Exploitation*](https://www.cs.cmu.edu/~sandholm/safeExploitation.ec12.pdf): motivates caution with opponent estimates. Our adaptation has no formal safety guarantee.
- [Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/): public no-limit betting and tournament convention reference for our simulator; no claim these are the contest's unseen engine details.
- [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html): offline model fitting/export API.
