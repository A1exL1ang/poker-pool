# Frozen candidate measurements

The candidate is the **24-tree default policy**, with the safe ICM premium-raise
guard. Runtime hashes are in `candidate.json` and `heldout_summary.json`.
All held-out scenario runs confirmed unchanged source and assets. No official
interface, actual finish-point vector, or full blind schedule was available.
These results concern this independent test environment only.

## Correctness and implementation

All 61 unit tests pass. Coverage includes minimum full raises; individual and
cumulative short all-ins and reopening rights; a prior check before a short
opening all-in; heads-up order and dead blinds; side pots; uncalled returns;
split pots and odd chips; simultaneous elimination; private observations;
legal raise-to amounts; all-in blind calls; inaccessible overbets; exact board
chops; hero-owned royal flushes; malformed-input fallback; per-table reset;
real model traversal; and no file or network access during `.act()`.

The original Python evaluator matches all **2,598,960 five-card category
counts**. A final run also matched 10,000 random seven-card hands to the best
of all 21 independent five-card evaluations. Final measured mean evaluation
time was 2.386 microseconds across 100,000 seven-card hands on Windows/Python
3.11.9. The original C++ generator separately passes exhaustive reference
checking and 1,500 cross-language score comparisons. The simulator helper's
500-table stress test conserved chips over 53,127 hands and 295,319 decisions,
with no errors or illegal baseline actions.

ICM utility agrees with complete finish-order enumeration through eight seats,
and with an independent subset dynamic program in 704 additional cases. Its
optimized subset formula is exact for the ICM model; the policy's choice of
hypothetical future pot outcomes remains an approximation.

## Training, development and final test separation

The original 64-tree model was fit on 600,000 states with 2,048 rollouts each,
split equally among flop, turn and river. The 169 preflop classes each received
400,000 rollouts, with seven independently defined range profiles. The fixed
range-ranking calculation uses seed 1100400 and 15,000 rollouts/class.

A disjoint 30,000-state, 8,192-rollout validation set assessed model quality,
packing and tree budgets. It was used for selection and is **not** the final
test. Exact integer splits plus quantized leaves reduced the full model from
111,872,710 to 51,838,011 bytes. On validation, mean quantization change was
0.0001844 and maximum change 0.0019534. The selected 24-tree prefix is
19,638,433 bytes. Relative to 64 trees, each profile's validation MAE increased
by less than 0.00013, while latency fell substantially.

Policy development used six predeclared configurations across mixed, tight and
loose tables, 400 tables per cell, **7,200 tables total**. Seeds begin at
1105000, 1115000 and 1125000. The greatest pooled measured change was +0.005
chip share with paired standard error 0.00808, insufficient evidence to change
the default parameters. Other variants showed no convincing broad improvement.
This study used 32 trees; budget selection then retained the default policy
and reduced the model to 24 trees. Safe semantic fixes and the explicit-payout
premium guard preceded the freeze. See `policy_tuning.json` and
`model_budgets.json`; development results are not held-out results.

Only after freezing was the separate **15,000-state final equity test** read.
Its generator seed is 1198800 with 16,384 rollouts/state. Final MAE against
these Monte Carlo labels is 0.00789–0.01376 across ten profiles; uniform HU MAE
is 0.00807. RMSE is higher, 0.01467–0.02913, so individual errors can be much
larger than the mean. Labels have finite sampling noise, particularly for
rare joint ranges. The final set is included as `final_equity_test.bin`.

## Frozen held-out tournaments

`python -m tools.final_evaluation` ran seven predeclared scenarios, **7,250
tables total**, using four processes with latency enforcement disabled to keep
policy outcomes independent of scheduling. No runtime source or asset changed.
Every scenario recorded zero policy errors, engine errors and illegal actions.
Approximate 95% confidence intervals below use sample mean +/-1.96 standard
errors. They describe sampling uncertainty, not simulator/model uncertainty.

- Mixed baselines, 1,500 tables, seeds 1190000+: mean final chip share
  **0.3480**, interval **[0.3239, 0.3721]**; mean finish rank 2.970.
- Tight baselines, 1,500 tables, seeds 1193000+: chip share **0.3360**,
  interval **[0.3121, 0.3599]**; mean finish rank 3.685.
- Loose baselines, 1,500 tables, seeds 1196000+: chip share **0.3013**,
  interval **[0.2781, 0.3246]**; mean finish rank 2.134.
- Six identical policy copies plus one TAG baseline, 500 tables, seeds
  1200000+: chip share **0.1580**, interval **[0.1260, 0.1900]**. This is a
  self-play sanity check, not evidence of superiority over identical agents.
- Hypothetical top-three payouts `[0.5,0.3,0.2,0,0,0,0,0]`, mixed opponents,
  1,000 tables, seeds 1202000+: mean points **0.2405**, interval
  **[0.2273, 0.2537]**; chip share 0.3450.
- Hypothetical linear payouts `[7,6,5,4,3,2,1,0]/28`, mixed opponents,
  500 tables, seeds 1204000+: mean points **0.17743**, interval
  **[0.17154, 0.18332]**; chip share 0.3320.
- Hypothetical 1.5x blind growth, mixed opponents, 750 tables, seeds 1206000+:
  chip share **0.33976**, interval **[0.30897, 0.37055]**. Capped surviving
  tables use ICM when an explicit payout vector is present.

Chip share is terminal chips /16,000, including fractional outcomes at the
200-hand cap. It is not a claim about official points or necessarily the same
as winning probability. Baselines are transparent and simple; no tournament
against unknown contest entrants was performed. Reports preserve every table's
seed and outcome in `heldout_*.json`.

## One-millisecond timing

Final commands use 20,000 realistic observations, a separate 2,000-state
eight-alive-seat ICM stress corpus, and seed 1300000. Timing includes the full
`.act()` call: validation, state, features, inference, ICM, policy and legalizing.
Asset loading and JSON transport are excluded and measured/described separately.
Every wall-clock outlier is retained. Both reports confirm unchanged source.

Windows/Python 3.11.9 (`latency_final_windows.json`):

- Cold fresh-state decisions: p99 **377.8 us**, p99.9 **607.1 us**, maximum
  **4,949.9 us**; **8/20,000 exceeded 1 ms**.
- Sequential table replay: p99 **417.8 us**, p99.9 **965.8 us**, maximum
  **2,924.6 us**; **19/20,000 exceeded 1 ms**.
- Warm repeat: p99 **97.6 us**, **0/20,000 exceeded 1 ms**.
- Eight-seat ICM stress: cold/warm/sequential overruns **2/0/1 per 2,000**.
- Enforced tournament capture: **13 timeout check/fold fallbacks across
  112,512 decisions by all eight agents**. Zero errors/illegal actions.
- Asset loading **234.8 ms**, outside the decision clock.

Linux/WSL Ubuntu 24.04/Python 3.12.3 (`latency_final_linux.json`):

- Cold fresh-state decisions: p99 **458.2 us**, p99.9 **886.8 us**, maximum
  **2,708.1 us**; **16/20,000 exceeded 1 ms**.
- Sequential table replay: p99 **397.4 us**, p99.9 **725.9 us**, maximum
  **5,486.2 us**; **6/20,000 exceeded 1 ms**.
- Warm repeat: p99 **126.9 us**, **2/20,000 exceeded 1 ms**.
- Eight-seat ICM stress: cold/warm/sequential overruns **2/0/3 per 2,000**.
- Enforced tournament capture: **30 timeout check/fold fallbacks across
  112,882 decisions by all eight agents**. Zero errors/illegal actions.
- Asset loading **632.8 ms**, outside the decision clock.

These are measured results, **not a hard 1 ms guarantee** on the official
hardware. The initial 64-tree candidate had 224/10,000 cold overruns and was
rejected in favor of the smaller candidate. Rare outliers also occur on warm
decisions; further tree reduction cannot establish a wall-clock guarantee.
Host-enforced fallback behavior is tested and retained in the simulator.

## Packaging and limits

`bot_11_offline.zip` contains Python source, both runtime assets, rules, README,
tests, local benchmark/training tools, measurements and final equity test data.
It excludes `.control`, credentials, caches, experimental model checkpoints,
native executables and remote access metadata. Gameplay needs only standard
Python; NumPy/scikit-learn are for training tools. See
`runtime_manifest.json` for runtime file hashes and the accompanying workspace
file `package_validation.json` for the independent extracted-package smoke
test (kept outside the archive to avoid a self-referencing archive hash).
Extracted-package checks use isolated Python with site packages disabled,
block network startup/play and all decision-time file I/O, and validate 200
offline actions plus JSON-lines transport. They run 58 passing tests and skip
the three training-generator integration tests because native binaries are
deliberately excluded. Those three pass in the full workspace after compiling
the included original C++ source; all 61 workspace tests passed.

The local contract requires initialization before the decision budget and a
fresh agent/reset at table boundaries. An official interface adapter remains
necessary. Missing actual payouts and blind amounts limit tournament-specific
optimization. Opponent-range, side-pot-equity, realization and shove models are
heuristics, not an equilibrium solution. Further tuning against these simple
local styles risks overfitting; the frozen candidate preserves a reproducible,
well-tested result without claiming absolute optimality.
