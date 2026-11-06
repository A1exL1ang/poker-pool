# bot_9 — independent local no-limit Hold'em bot

This is an original, offline Python implementation for **local experimentation**.
It is **not submitted**, and it does **not claim compatibility with an official
Agent interface**. No official starter kit, payout table, or complete blind
schedule was available. Read [RULES.txt](RULES.txt) for the supplied contest rules.

The collectible runtime is `bot.Agent`, also exported as `bot.PokerBot`. Create
one instance per table, then call `agent.act(observation)`. The runtime uses only
Python's standard library and the included assets. It performs no live model
calls, sampling, network access, subprocess execution, or training.

## Run it

```powershell
python -m unittest discover -s tests -v
python tools/benchmark.py --tables 100 --split tuning --lineup mixed --out measurements/local_tuning.json
python tools/latency.py --calls 20000 --out measurements/local_latency.json
python tools/package_bot.py
```

`python play.py` is an optional **local JSON-lines runner**, not an official
adapter. Send an observation per line; send `{"reset":true,"seed":9}` to reset
table state. Parsing, printing, imports, and construction are outside the
strategic `act()` timing measurement. An actual contest adapter must be written
and verified against the real starter kit when it becomes available.

## Local observation and action contract

Card IDs are `4 * (rank - 2) + suit`: ranks 2–14, suits 0–3. There are 52 cards.
An observation is a plain dictionary with these fields:

- `hero`, `button`: seat indices; arrays retain empty seats.
- `hole`: the hero's two cards; `board`: zero, three, four, or five public cards.
- `stacks`: chips **behind**, excluding all commitments.
- `committed`: each seat's contribution on the **current street**.
- `contributions`: each seat's contribution over the **whole hand**.
- `folded`, `all_in`: per-seat booleans.
- `street`: 0 preflop, 1 flop, 2 turn, 3 river; `hand_no`: table-local integer.
- `pot`: sum of whole-hand contributions, including current-street wagers.
- `to_call`: additional chips required from hero, capped at hero's stack.
- `min_raise_to`: minimum full raise's **total current-street commitment**.
- `max_raise_to`: hero's entire stack plus current-street commitment.
- `can_raise`: true only if action is reopened and someone can contest a raise.
- `bb`: current big blind.
- `history`: public actions, each with `seat`, `street`, `action`, and `amount`.
  A raise's amount is the total street raise-to; the other amounts are actual
  chips paid. Blind posts are represented in commitments, not action history.
- Optional `payouts`: explicit, descending rank-order points padded to all seats.
  If absent, the bot applies no invented payout-related risk premium.

Actions are `{"action":"fold"}`, `{"action":"check"}`, `{"action":"call"}`,
or `{"action":"raise","amount":TOTAL_STREET_COMMITMENT}`. A short all-in can
be below `min_raise_to` only when the target is `max_raise_to`. The bot clamps
raises to these limits; the simulator separately validates every action.

The simulator `engine.Hand` and `engine.tournament` provide complete examples.
Observations contain no opponent private cards or future deck information.
Opponent statistics use anonymous seat actions observed at this table only.

## Strategy and training

Preflop play combines position, effective stacks, a combination-weighted hand
percentile, pot odds, shallow-stack shoves, and a reproducible 169-hand lookup
against four range priors. Each lookup cell uses 50,000 independent runouts
(seed 901). Opponent range priors are complete hand types ranked by random
heads-up equity, not solver ranges. Actual masses are 100%, 536/1326, 208/1326,
and 68/1326 combinations; interpolation below the tightest range is clamped.

Postflop play uses a compact 12-tree regression forest over 84 suit-invariant
features. Its six targets are heads-up equity against random holdings, the
three preflop priors, and top 40%/15% **current made-hand** priors. Whole boundary
rank ties are included, so made-range masses can exceed the nominal percentage.
Legal opponents and runouts exclude the hero's cards and board. Betting ranges
mix these value estimates with a conservative bluff prior. This is a heuristic
range model, **not CFR, a Nash strategy, or a solved multiplayer game**.

The final model uses 420,000 newly generated positions, 384 runouts per target,
from seeds 193000–193006. Architecture validation uses 18,000 positions,
1,024 runouts, seed 825123. A separate 12,000-position, 2,048-runout test uses
seed 9371821; the frozen model's final test MAE is 1.25–1.53 percentage points
across the six targets, measured against Monte Carlo labels. Training uses
NumPy/scikit-learn; gameplay does not import either. The 24-tree training
forest is exported to 12 trees for runtime speed. Integer-valued features allow
flooring split thresholds without changing branches; float32 leaves introduce
only small numerical rounding. Binary-vs-JSON parity tests cover both schemas.

Calls account only for pots the hero can contest. Supplied payouts trigger an
exact binary-outcome ICM calculation with at most five survivors in an eligible
heads-up pot, and a bounded approximation otherwise. Multiway equity uses an
explicit power approximation. Shrunk table-local action frequencies adjust
range and bluff priors. Neither the ICM outcome model nor range inference is exact.

## Rules, simulator assumptions, and tests

The local benchmark uses eight seats, 2,000 chips, blind changes every 25 hands,
and a 200-hand cap. The supplied rules specify the initial 10/20 blinds, but not
later levels or points. Consequently all benchmark payout presets and later
blind levels are **synthetic**, passed explicitly and recorded in each result.
Presets include winner-take-all, `[.5,.3,.2,0,0,0,0,0]`, and flat top three.
None is attributed to the Organizer.

Additional local assumptions: no antes/rake, a uniformly shuffled deck without
burns, one board, a moving live button, odd chips clockwise left of the button,
and tied simultaneous busts splitting their finishing payouts. Survivors at the
hand cap receive exact ICM points. An official simulator may differ.

Tests cover full/minimum/short raises and cumulative reopening, heads-up order,
BB options, atomic invalid-action rejection, uncalled returns, side pots, odd
chips, ties, all-in runouts, privacy, chip conservation, ICM, and safe fallbacks.
The evaluator matched an independent reference on all 2,598,960 five-card hands,
plus randomized five-/six-/seven-card comparisons. A separate property audit
replayed 2,000 hands and 11,417 actions against independent settlement and
suit-permuted states, with zero failures. Reproduce the expensive evaluator
check with `python tools/validate_evaluator.py` (this always runs the full check).

## Reproducible experiments

All outputs in `measurements/` describe **local synthetic** games. Tuning seeds
are below 10000; tournament holdout seeds begin at 100000. Normal-approximation
confidence intervals describe variation across independently seeded tables;
they do not establish performance against unknown contest opponents.

Policy selection used 4,896 first-stage and 19,584 second-stage tournaments.
The selected `tight_low_bluff` policy was within 0.00183 average normalized
points of the highest-scoring second-stage variant; the predeclared 0.005 tie
rule preferred its lower bluff frequency. Its point estimates exceeded the
initial parameter reference in all five second-stage suites. This is tuning
evidence; final held-out results are separate. `--lineup reference` freezes this
bot's initial eleven parameter values with the same current model/source;
`--lineup mirror` uses the final policy itself.

The frozen policy was then evaluated on **6,320 held-out tables**, with no
subsequent policy or model tuning. Across 1,054,373 focal decisions there were
zero engine-invalid actions and zero internal safety fallbacks. Of these,
1,000,378 decisions occurred in 6,000 timing-enforced tables, with zero focal
timeouts. Five timeouts occurred among other players over 5,646,372 total
decisions; these are retained in the raw diagnostics.

Selected held-out point estimates and 95% intervals:

- Seven copies of the initial parameter reference, 1,000 tables:
  **0.15536 [0.14375, 0.16697]**, versus equal share 0.125.
- Mixed heuristic pool with synthetic SNG payouts, 1,000 tables:
  **0.32064 [0.31025, 0.33104]**.
- Mixed pool with winner-take-all payouts, 1,000 tables:
  **0.40360 [0.37495, 0.43224]**.
- Seven slow Monte Carlo opponents, only 64 tables:
  **0.17540 [0.13317, 0.21763]**; this is a small, heuristic-opponent sample.
- Self-play, 1,000 tables: **0.10943 [0.09875, 0.12011]**, below equal share
  0.125. This unexpected sample is preserved. A separate four-seed diagnostic
  found identical trajectories under different Agent construction orders and
  no state-sharing asymmetry; it does not rule out every possible issue.

Flat-top-three and faster-blind sensitivity results, full finishing
distributions, source/asset hashes, runtime versions, and every per-table result
are in `measurements/final_evaluation.json` and `measurements/final_*.json`.
These results support the local implementation; they do not predict an unknown
contest field or establish absolute optimality.

```bash
g++ -std=c++17 -O3 tools/generate_equity.cpp -o generate_equity
./generate_equity validate
./generate_equity validate-made
./generate_equity preflop assets/preflop.csv 50000 901
./generate_equity samples6 train.csv 60000 384 193000 assets/preflop.csv
# Repeat for the seven training seeds; generate separate validation/test files.
python tools/train_equity.py --data data6/train_*.csv --holdout data6/validation.csv --out assets/equity6_model.json.gz --trees 24 --depth 18 --leaf 8
python tools/compact_model.py --source assets/equity6_model.json.gz --trees 12 --holdout data6/validation.csv --rows 18000
python tools/tune.py --tables 128 --workers 8
python tools/benchmark.py --tables 1000 --split heldout --lineup mixed --payouts sng --enforce-time-limit --out measurements/new_holdout.json
```

Do not reuse final holdout seeds to tune and still call that result held-out.
The `mc` opponent is a deliberately slower, 96-rollout benchmark policy; its
tables disable timing enforcement to avoid manufacturing wins from its timeouts.
Mirrors and heuristic opponents support timing-enforced tests separately.

The bounded Slurm scripts in `tools/` reproduce the team's own remote jobs when
placed in its authorized private directory. Resource accounting lives in
`.control/jobs.json`. All pre-existing user jobs were left untouched.

## Runtime and limitations

The package tool builds `dist/bot_9.zip` from a whitelist, records SHA256 hashes,
enforces the 100 MB limit on both ZIP and unpacked contents, and runs isolated
`python -I -S` decisions with socket operations blocked and no third-party
modules. Training files, credentials, control metadata, generators, executables,
and benchmark opponents are excluded.

**A hard 1 ms guarantee has not been established.** The compact predictor's
Linux validation ran 18,000 calls with zero wall-clock overruns (median about
62 µs, p99 about 74 µs), but full-agent calls and Windows measurements can have
OS scheduling/GC outliers. Reports preserve every measured overrun, including
automatic check/fold fallbacks in timing-enforced tournaments. Construction and
import timing is reported separately; the actual platform's accounting is
unknown. Windows process-time clocks are too coarse for reliable per-call CPU
timing. Final full-agent results are recorded alongside the package manifest.

Final dedicated **Linux** replay: 40,000 full `act()` calls, **two wall-clock
overruns** (0.005%), p99 approximately **0.103 ms**, maximum **1.615 ms**. Both
outliers also exceeded 1 ms on the process clock. Final **Windows** replay:
20,000 calls, **five wall-clock overruns** (0.025%), p99 **0.209–0.238 ms**,
maximum **72.972 ms**. No invalid actions, internal fallbacks, exceptions, or
stateful replay mismatches occurred. The original timing outliers are not
removed or attributed to a single cause. Linux cold import/construction took
about 27.2/8.5 ms; Windows about 53.5/12.7 ms, measured outside `act()`.

The final local ZIP is **5,302,616 bytes compressed**, **15,008,822 bytes
unpacked**, with SHA256
`2b800e5c57fc7e963a66e7cf2f05efe2e587e5e071e56ea90e4f19d850faf3f1`.
Windows and Linux both passed all 50 tests and isolated package smoke checks.
ZIP host metadata can make archive hashes differ by platform; the frozen
runtime source and model hashes matched between machines.

Other material limitations: no official interface testing; unknown actual
payouts/blinds/opponent distribution; heuristic action abstraction, bluffing,
range inference and multiway equity; no equilibrium/exploitability guarantee;
no claim of contest optimality. Further work should start with the real kit's
adapter, then repeat legality, timing and payout-aware tests on that platform.

Public research consulted for strategy design (no external poker code copied):
[Brown & Sandholm, multiplayer poker](https://noambrown.github.io/papers/19-Science-Superhuman.pdf),
[Libratus](https://ijcai.org/proceedings/2017/772),
[PokerStars on tournament risk](https://www.pokerstars.com/poker/learn/lesson/icm-and-its-impact-on-tournaments/),
and [Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/).
