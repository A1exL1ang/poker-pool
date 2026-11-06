# bot_11: independent offline no-limit Hold'em bot

This is a locally developed candidate and test environment, **not a contest
submission**. No official starter kit was available. The observation schema,
Python entry point, simulator, later blind levels, and hypothetical test payouts
below are local assumptions. An adapter to the actual official Agent interface
will be necessary when that interface is supplied. Nothing was pushed,
published, or submitted.

## Run

Python 3.11+; **no third-party runtime dependencies**, network, subprocesses,
online tools, or gameplay-time file reads. All assets are included.

```python
from agent import Agent
bot = Agent(seed=110011)    # load assets BEFORE the decision clock starts
action = bot.act(observation)
```

Construct a new instance at every table, or call `reset()`. Opponent information
is limited to anonymous seats at the current table. Shared equity assets are
read-only. `python agent.py` also accepts one local observation JSON object per
line and writes an action JSON object per line. JSON parsing/serialization and
process startup are **not** included in `.act()` latency measurements; that
command is a smoke-test transport, not a 1 ms process-per-action protocol.

## Local observation and action semantics

`observation` is a dictionary with these required keys:

- `hole_cards`: two distinct integer cards; `board`: zero/three/four/five cards.
  Card ID is `4 * rank + suit`, with rank 0=2 through 12=A, suits 0=clubs,
  1=diamonds, 2=hearts, 3=spades. Rank/suit strings such as `As` also work.
- `seat`, `button`, `hand_number`, `small_blind`, `big_blind`, `street`.
  Streets are exactly `preflop`, `flop`, `turn`, `river`.
- `players`: a list containing **all table seats**, including busted seats.
  Each player has `seat`, `stack` (remaining uncommitted chips), `street_bet`
  (already committed this street), `committed` (already committed this hand,
  including `street_bet`), `folded`, `all_in`. Seat IDs are consecutive integers
  from zero; list order matches seat order. No opponent hole cards are accepted
  or needed by the policy.
- `pot`: all committed chips, including outstanding bets and folded players'
  contributions. `to_call`: **additional chips** needed to call, capped by
  hero's remaining stack. `current_bet`: the highest street commitment.
- `min_raise_to`, `max_raise_to`: **total street commitments** after raising.
  `max_raise_to = hero.street_bet + hero.stack`. `can_raise` and
  `legal_actions` describe actual action rights, including short all-ins that
  do not reopen betting. `legal_actions` uses `fold/check/call/raise`.

Optional keys:

- `small_blind_seat`, `big_blind_seat`: exact blind positions; include these for
  dead-button/dead-small-blind accuracy. The simulator supplies them.
- `history`: chronological actions from the **current hand only**. Each has
  `seat`, `street`, `action`, and `amount` (raise-to for raises, added chips for
  calls/blinds). Full action history improves range and opponent estimates.
- `payouts`: actual finish-point vector, first to last, covering all original
  seats. Omit it when unknown. It must be finite and nonincreasing. Actual
  contest payout points were not supplied. Regional/finals cash prizes are
  never treated as table payouts.

The returned dictionary is `{'action': 'fold'}`, `check`, `call`, or
`{'action': 'raise', 'amount': INTEGER_RAISE_TO}`. Bets use `raise` with the same
raise-to convention. Raises are clamped to legal bounds; a short all-in may
land below the normal full-raise minimum when the maximum permits only that.
Errors produce check if available, otherwise fold, and increment
`bot.diagnostics['errors']`. The caller remains responsible for enforcing the
platform's own deadline. The independent simulator can enforce 1 ms wall time
and records every resulting fallback.

## Strategy

All poker code and assets were developed independently in this workspace.

- Preflop: original Monte Carlo equity estimates for all 169 starting-hand
  classes against seven strength-ranked ranges; position-dependent opens,
  isolation raises, value three-bets and a small suited-wheel-ace bluff range;
  stack/pot-odds-aware calls and a short-stack shove EV approximation.
- Postflop: 58 suit-invariant features encode the exact made-hand category and
  kickers, ownership of pairs/sets, board texture, flush and straight draws,
  and blockers. A 24-tree multi-output forest predicts ten equity profiles,
  including one/two/three uniform opponents, three preflop ranges, two current
  made-hand ranges, and two joint ranges. Runtime inference is pure Python.
- Betting history estimates per-opponent ranges. Anonymous seat statistics use
  conservative priors and only observed current-table actions. Value sizing,
  selective semi-bluffs, limited bluffs, and equity realization discounts
  depend on position, street, opponents, pot size, and effective stack.
- Multiway equity uses independently learned uniform multiway moments to
  correct the product approximation. Calling accounts for eligible side-pot
  layers and inaccessible overbets. Universal board chops and hero-owned royal
  flushes have exact overrides.
- When a real payout vector is supplied, large heads-up calls can use exact
  ICM utility and an approximate binary pot-resolution model. A subset-sum
  exponential-race formula avoids expensive finish-order enumeration. Missing
  payouts select chip-EV behavior, with no invented prize structure.

This is an equity-guided policy, not a solved eight-player equilibrium or a
Pluribus reproduction. Current-range profiles and multiway corrections are
approximations; postflop draws are partly represented by the learned equity
features, rather than a complete Bayesian range tree. The short-stack shove
model approximates caller behavior and multi-caller outcomes. ICM fold/win/loss
resolution is used only with one live opponent, no inaccessible side pot, and
a large call. It does not solve future tournament play or include future skill.

## Rules and simulator assumptions

The complete supplied `RULES.txt` was read. Relevant constraints: eight bots,
2,000 starting chips, initial 10/20 blinds, levels every 25 hands, 200-hand cap,
ICM allocation of remaining points, offline decisions within 1 ms, 100 MB file
limit, and table-boundary state resets. The six tournaments have weights 1..6.

`pokerbot.engine` implements an independent standard NLHE simulator with full
and short raises, cumulative reopening, all-in runouts, side pots, uncalled
returns, split pots/odd chips, dead blinds and heads-up transitions. Its default
blind schedule **assumes doubling** every 25 hands because later official blind
amounts are missing. A 1.5x alternative is included in benchmarks. No antes or
rake are assumed. Defaults supply **no payout vector**. Optional winner-take-all,
top-three and linear finish rewards in benchmark scripts are clearly synthetic
sensitivity scenarios; none is represented as official.

Mechanics research used the primary [Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/),
particularly minimum full raises, prior-check action rights, dead blinds and
heads-up button placement. The actual contest engine takes precedence when
available. Equity/range design was informed by the primary University of
Alberta paper [The challenge of poker](https://poker.cs.ualberta.ca/publications/AIJ02.pdf)
and Brown/Sandholm's [multiplayer poker research](https://doi.org/10.1126/science.aay2400).
No external poker implementation or private repository was used.

## Training and reproduction

Training requires C++17/OpenMP, NumPy and scikit-learn; gameplay does not.

```text
g++ -O3 -std=c++17 -fopenmp tools/generate_equity.cpp -o generate_equity
generate_equity --mode postflop --states 600000 --rollouts 2048 --seed 1101201 --threads 8 --out train.bin
generate_equity --mode postflop --states 30000 --rollouts 8192 --seed 1102202 --threads 8 --out holdout.bin
generate_equity --mode preflop --rollouts 400000 --seed 1103303 --threads 8 --out assets/preflop.json
python -m tools.train_equity --train train.bin --test holdout.bin --trees 64 --leaf 6 --depth 23 --jobs 8
python -m tools.compact_equity
python -m tools.model_budget --source assets/equity_compact.bin.z --trees 24 --export-only
```

Replace the full-precision asset with the resulting `assets/equity_24.bin.z`.
The fixed independent range-ranking seed is 1100400 with 15,000 uniform
rollouts/class. All ten target definitions, fractional range-boundary weights,
binary record layout and seeding are in `tools/generate_equity.cpp`. Training
labels are simulations, not exact equity. The independent validation generator
uses disjoint state/runout seeds; its finite Monte Carlo noise contributes to
reported prediction errors. This 30,000-state set was used to assess numerical
packing and choose the tree budget, so it is validation, not an untouched final
test. Tournament policy tuning uses separate development seed blocks. After
freezing the 24-tree default policy, a third 15,000-state set (seed 1198800,
16,384 rollouts/state) was evaluated once; it is included as
`benchmarks/final_equity_test.bin` for reproducibility.

Integer split thresholds preserve every tree branch because all 58 features
are integers. Leaf quantization introduces at most 1/510 absolute error per
tree/output. On the 30,000 validation states, observed mean output change is
0.0001844 and maximum 0.0019534. The full 64-tree compact asset was 51,838,011
bytes versus 111,872,710 bytes for the original compressed full-precision
model. The selected 24-tree asset is **19,638,433 bytes**. Dropping from 64 to
24 trees changed mean absolute errors by less than 0.00013 per profile on
validation while substantially reducing latency. Final independent-test MAE
spans **0.00789–0.01376** across profiles. Detailed per-profile/street metrics,
seeds, parameters and hashes are in `benchmarks/equity_model.json`,
`benchmarks/equity_compaction.json`, `benchmarks/model_budgets.json`, and
`benchmarks/equity_final_test.json`.

`tools/run_training.sh` and `tools/run_compaction.sh` record the bounded private
compute recipe. Raw generation files remain in the authorized private remote
directory; runtime assets and source are local. Only newly requested jobs were
used; no pre-existing allocation was inspected, attached to, or modified.

## Validation commands

Run from the workspace root:

```text
python -m unittest discover -s tests -v
python -m tools.evaluator_check --hands 100000 --cross-check 10000 --exhaustive
python -m tools.benchmark --tables 500 --seed 1104000 --ecosystem mixed --out benchmarks/development.json
python -m tools.benchmark --tables 1000 --seed 1190000 --ecosystem mixed --out benchmarks/heldout_mixed.json
python -m tools.final_evaluation
python -m tools.evaluate_equity --data benchmarks/final_equity_test.bin
python -m tools.latency --samples 20000 --stress-samples 2000 --seed 1300000 --enforce --out benchmarks/latency_reproduction.json
python -m tools.package
python -m tools.offline_check
```

Measurements, frozen policy hashes, timing overruns, held-out results, and
limitations are recorded in `benchmarks/RESULTS.md`.
Simple local opponents and an independent simulator cannot establish contest
optimality or predict results against unknown teams. Timing depends on CPU,
Python version, scheduling and the official budget boundary; wall-time outliers
are retained and reported, not silently discarded.
