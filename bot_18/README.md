# bot_18 — independent offline Hold'em bot

This is a local experimental implementation of standard no-limit Texas Hold'em.
The entire supplied `RULES.txt` was reviewed. There is no official starter kit,
Agent API, payout vector or complete blind schedule in this workspace. **This
package does not claim official interface compatibility and has not been
submitted, pushed or published.** An official adapter must be written and tested
when the real interface becomes available.

The runtime is pure Python standard library, with offline-trained equity assets.
Python 3.10+ is required; validation targets Python 3.11 and 3.12. Gameplay makes
no network calls, runs no external processes and performs no Monte Carlo rollouts.
Import/model initialization is outside the measured `Agent.act` decision path.

```python
from agent import Agent
bot = Agent(seed=18)         # supply seed for reproducibility; default uses local entropy
action = bot.act(observation)
bot.reset()                 # required if reusing an instance at a new table
```

`run_local.py` also offers a JSON-lines adapter for the **local** contract. Its
serialization and process startup are not included in decision-time measurements.

## Local observation and action contract

See `sim/schema.py` for the full schema and `sim/README.md` for engine semantics.
Observations are dictionaries containing:

| Field | Meaning |
|---|---|
| `hand_id`, `street` | Table-local hand index; `preflop`, `flop`, `turn`, `river` |
| `hero`, `button` | Physical seat IDs; at most eight seats in bot policy |
| `hole`, `board` | Card integers 0–51: rank `card//4+2`, suit `card%4` |
| `players` | All seats: `seat`, remaining `stack`, street `bet`, hand `invested`, `folded`, `all_in` |
| `pot` | All invested chips, including current-street bets |
| `to_call` | Additional chips required, capped by hero's remaining stack |
| `min_raise_to` | Minimum full raise's TOTAL street contribution |
| `max_raise_to` | Hero's remaining stack plus existing street bet |
| `can_raise` | Whether raising is legal, including reopening and live responders |
| `history` | Current hand's public actions: seat/street/kind/amount/raise_to; no opponent hole cards |
| `small_blind`, `big_blind` | Current blind amounts |
| `payouts` | Optional supplied finishing-point vector, best place first; absent means chip EV |
| `hands_remaining` | Optional number of hands before the local cap |

Return `{'action':'fold'}`, `{'action':'check'}`, `{'action':'call'}`, or
`{'action':'raise','amount':N}`. A raise's amount is **total current-street chips**,
not an increment and not a whole-hand contribution. A short all-in may be below
`min_raise_to` only when it equals `max_raise_to`. A check is used whenever no call
is due. Malformed observations use check/fold fallback and increment `Agent.errors`.

## Strategy and assumptions

- Position-dependent opening/isolating ranges and measured preflop equity tables.
- Short-stack shove expected-value approximation, with all-in opponents excluded
  from fold-equity assumptions; premium tables for strong re-raising ranges.
- A compact boosted-tree model estimates showdown pot share against a documented
  preflop range abstraction, including blockers and multiple opponents.
- Public betting history and shrinkage-based opponent statistics adjust ranges,
  postflop selection, bluff frequency and value betting. Opponent state is kept
  within one table and reset between tables; no team identity is used.
- Exact ICM evaluates selected heads-up terminal calls when actual
  payouts are supplied. Its generated subset-moment implementation agrees with
  the readable subset-DP reference to floating-point precision. An optional
  short-stack shove calculation uses exact ICM at approximate one-caller leaves.
  The win/lose future-state model is an approximation;
  guaranteed board chops bypass it. Missing payouts do not invent a tournament
  distribution. Multiway side-pot strategy uses conservative approximations.

This is an experimentally tuned policy, **not an equilibrium solver**. Its
opponent range and equity-realization assumptions can be exploited. The equity
model estimates showdown equity, not the value of a complete future betting
strategy. See `equity/README.md` for model details and approximation limits.

Selected settings are opening-range scale 1.2, postflop selection adjustment
1.5, bluff-frequency scale 1.6, call margin 0.025, and payout-aware short-stack
shoves enabled when supplied payouts and the modeled conditions permit them.
The last shove comparison was statistically inconclusive (pooled synthetic
means 26.555 versus 26.542); the selected switch follows the recorded selection
rule, not a claim of a demonstrated performance gain. Earlier tuning favored
wider opens and stronger adjustment for opponents' betting selection.

## Reproduce locally

```text
python -m unittest discover -s tests -v
python -m sim.benchmark --candidate agent:Agent --tables 100 --seed 182000 --suite mixed --payout top3 --out measurements/reproduction.json
python training/validate.py --native training/generate.exe --samples 20000
python build_package.py
```

Training requires NumPy, scikit-learn and a C++17/OpenMP compiler. These are
development dependencies only. `training/generate.cpp`, `training/fit.py` and
`training/experiments.py` reproduce data generation, model fitting and policy
experiments. Preserve seeds and distinguish tuning from independent evaluation.
Three model versions and four policy-tuning rounds are retained so the selected
candidate is reviewable. `checkpoints/` contains earlier runnable runtime ZIPs.
Job commands and own-job ledger are in `.control`; no remote resource is needed
to use the packaged bot. Large raw training CSVs may remain in the bot's private
remote directory; source, compact assets and measurement reports are local.

All benchmark payout vectors and later blind schedules are explicitly synthetic.
The supplied rules establish eight players, 2,000 starting chips, initial 10/20
blinds, increases every 25 hands, a 200-hand cap and ICM allocation at the cap.
Local schedules after the initial level and point distributions are assumptions.

## Evidence and limitations

Final measurements are recorded under `measurements/` and
`training/final_evaluation.json`; versioned `training/metrics*.json` files
describe development validation only.
Reports distinguish model label validation, policy tuning, held-out tournaments,
decision latency and packaging. Local opponents are independently written
heuristics; performance against them is not evidence of optimality or a forecast
of contest rank. Timing is hardware/OS-dependent; wall-clock pauses count as
overruns and are reported rather than discarded. The contest's handling of
startup, adapter overhead and sandbox scheduling remains unknown.

The original development split called `heldout.csv` in training commands was
used to diagnose and refine the model; it is **validation**, not final test data.
The final model test uses a distinct frozen seed and includes opponent counts
4 and 6, which were absent from the postflop training grid. The first frozen
evaluation exposed that tree thresholds aliased those counts to 3 and 5,
respectively. The corrected runtime interpolates log pot-share between the
supported counts; exact tie and rare-board formulas remain exact. That first
evaluation is retained as a development audit, and the corrected runtime uses
fresh final seeds. Final tournament
seeds and the reserved `challenge` opponent population are separate from tuning.
The `selfplay` suite consists solely of independently seeded variants of this
same bot; it does not involve other teams or external/private poker code.

After the quiet-laptop instruction arrived, all substantial remaining builds,
tests and experiments were moved to new bounded private Engaging jobs. No
pre-existing allocation, node session, job or shared SSH master was touched.

## Public references consulted

- [Supplied contest rules source](https://pokerbot.sequoiacap.com/rules.html);
  the local `RULES.txt` is the task's rule source.
- [Poker TDA 2026 rules](https://www.pokertda.com/view-poker-tda-rules/) informed
  local betting, reopening, side-pot and odd-chip mechanics.
- [GTO Wizard: ICM Basics](https://blog.gtowizard.com/icm-basics/) explains why
  supplied payouts affect tournament risk and why ICM has limitations.
- [Zinkevich et al., Regret Minimization in Games with Incomplete Information](https://proceedings.neurips.cc/paper_files/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf)
  and [Brown et al., Deep CFR](https://proceedings.mlr.press/v97/brown19b) informed
  the distinction between equity estimation and equilibrium strategy solving.

Implementation and training data were created independently in this workspace;
no other bot, private poker repository or other team's artifacts were inspected.

## Final frozen-candidate measurements

The frozen candidate completed **13,824 held-out local tournaments** and **2,558,316 decisions** with the 1 ms fallback enforced. There were 0 external exceptions, 0 illegal actions, 0 internal error fallbacks and **12 timing overruns** (0.00047%). Overruns became checks/folds in the simulator.

Separate Linux replay/stress timing (33,287 calls): median 31.3 µs, p99 214.1 µs, p99.9 239.4 µs, maximum 3571.1 µs; 1 calls exceeded 1 ms. Initialization and serialization are excluded. These are measured results, not a hard real-time guarantee.

The independent equity test used 28,000 fresh states with 8,192 Monte Carlo trials each: MAE 2.50 percentage points, RMSE 4.39 points. The 95th-percentile absolute error was 9.18 points and the largest was 48.59 points, so individual rare-state estimates can be substantially wrong. These targets use the documented range abstraction.

Standard-schedule synthetic 50/30/20/0/0/0/0/0 point results (512 tables per row):

| Opponent suite | Mean points | Approximate 95% interval |
|---|---:|---:|
| mixed | 32.19 | 30.66–33.71 |
| calling | 46.33 | 45.44–47.21 |
| tight | 26.55 | 24.76–28.34 |
| aggressive | 31.55 | 30.23–32.88 |
| challenge | 15.35 | 13.69–17.00 |
| selfplay | 12.02 | 10.58–13.47 |

Equal allocation of this synthetic point pool is 12.5 per seat. These opponents are local heuristics or variants of this bot; the figures do not establish contest rank or optimality. All payout/blind sensitivity cells and source hashes are in `measurements/final_summary.json`.
