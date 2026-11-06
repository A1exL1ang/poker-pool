# Equity estimation

The evaluator, generator, feature design, and inference runtime were written
independently in this workspace. XGBoost is used only to fit offline regression
trees. Gameplay uses Python's standard library and checked-in JSON assets.

Cards are integers `4 * (rank - 2) + suit`. Five-card ranks are packed into an
integer with category in bits 20 and above and five rank tiebreak digits in
bits 16, 12, 8, 4, and 0. Higher wins. Categories are high card, one pair, two
pair, trips, straight, flush, full house, quads, straight flush. Aces are high
except in an A2345 straight. The evaluator handles any five to seven cards;
the feature routine also uses padded three/four-card board ranks.

`poker_eval.FEATURE_NAMES` is the authoritative 43-feature schema. The native
generator and Python runtime have parity tests. `equity_model.predict_equity`
extracts features; `predict_from_features` accepts an existing feature vector
to avoid repeating extraction. These functions estimate heads-up showdown
pot share. They do not compute a betting equilibrium or a best response.

The refined fitter adds 12 cheap relational features in `equity_features.py`
without changing the 43-value input or regenerating native data. These expose
rank improvements over the board, counterfeiting, board pairs/trips, and the
gap between a made straight and possible higher straights. The model declares
its input schema, derived version, and full 55-feature model schema.

## Declared opponent-range model

First, each of 169 canonical starting-hand classes is simulated against a
uniform random holding. Classes are ordered by this Monte Carlo equity and
assigned a percentile using their actual 6/4/12 combination multiplicities.
This is a showdown ranking; it does not claim optimal preflop playability.

For `range_width=w<1`, a holding with percentile `p` gets prior weight
`1 / (1 + exp((p-w)/max(.012, .075*w)))`. Width 1 gives uniform weights.
The model and preflop table support widths from .05 through 1.0. Narrower
real ranges cannot be represented exactly by the policy's .05 floor.

For postflop `aggression=a`, the prior is multiplied by `c**(3*a)`, where `c`
is the explicit made-hand/draw continuation score in `native/equity.cpp`.
Zero aggression leaves only the preflop prior. One strongly favors made
hands and live draws. This is a hand-built range model, not measured behavior
of the contest field. It can misdescribe bluffs, traps, range construction,
and action histories; within-table policy adaptation remains heuristic.

Heads-up opponent combinations are fully enumerated and normalized after
removing hero cards and the observed board. River equity is exact under the
declared weights. On flop/turn the opponent is sampled from that distribution,
then the remaining board is sampled uniformly without replacement. Labels
therefore respect hole-card and board blockers. Flop/turn labels have Monte
Carlo noise. Validation and held-out labels use more draws than training.

The preflop table covers 169 classes, widths .05/.10/.20/.35/.60/1.0, and one
through seven opponents. Villains are sampled sequentially with card removal,
then the board is dealt; pot shares include ties. Sequential range removal is
an approximation to a fully normalized joint product of player ranges.

## Reproduction

Run from the bot_19 workspace or its authorized private remote directory:

```bash
g++ -O3 -std=c++17 -fopenmp native/equity.cpp -o native/equity
export OMP_NUM_THREADS=8
python3 training/validate_native.py ./native/equity
./native/equity base training/data/base.txt 100000 1900101
./native/equity preflop training/data/base.txt assets/preflop.json 12000 1900102
./native/equity generate training/data/base.txt training/data/train.bin 600000 192 1900103
./native/equity generate training/data/base.txt training/data/validation.bin 45000 1024 1900104
python3 training/train.py --train training/data/train.bin --valid training/data/validation.bin --trees 192 --depth 8 --threads 8
python3 training/export_fast.py
python3 training/validate_fast.py
./native/equity generate training/data/base.txt training/data/final_equity_test.bin 30000 8192 1900205
python3 training/evaluate_model.py --test training/data/final_equity_test.bin --report measurements/equity_final_heldout.json
```

Create `assets`, `measurements`, and `training/data` first. Fitting requires
NumPy and XGBoost, which can be installed into a private virtual environment.
The exact versions and data/model SHA256 hashes are recorded in the training
report. Native binary rows are 44 little-endian float32 values: 43 features
followed by showdown share. Row seeds are deterministic across thread counts.
`training/job.sh` is a bounded eight-CPU/32GB Slurm recipe, with a bot_19 name.
The root agent alone owns job submission/cleanup and its resource ledger.

Training uses seed 1900103; model choice uses seed 1900104 only. The original
prototype used held-out seed 1900105. After refinement using validation-only
diagnostics, the selected model was evaluated on fresh seed 1900205. The held-out
equity metric measures approximation of this synthetic range model, not
tournament win probability. Tournament validation is reported separately.

The selected model has 192 depth-eight trees per street and 55 model features.
`measurements/model_selection.json` records the preset validation rule and
the depth-six/depth-eight/256-tree comparisons. The 256-tree candidate improved
validation mean RMSE by 4.95%, below the preset 5% threshold for added size.
The chosen model's mean validation RMSE was 0.02536 versus the original model's
0.03447. Its model SHA256 is
`e19b864bc347f3a091feb1dbc91f3b7496585f9e935f6b674d79938f9c123944`.
`training/refine_job.sh` records the bounded refinement recipe. Fitting used
Python 3.12.7, NumPy 2.5.4, and XGBoost 3.4.1; none is a gameplay dependency
beyond a supported Python interpreter.

On the fresh 30,000-state held-out set, RMSE was 0.01623 on flop, 0.02276 on
turn, and 0.03159 on river. Maximum absolute errors remained 0.279, 0.255,
and 0.317 respectively: average calibration does not eliminate difficult
tails. The full metrics and input/model hashes are in
`measurements/equity_final_heldout.json`.

The standard-library export is verified against XGBoost predictions before
writing assets. Split thresholds and the two continuous inputs are normalized
to float32 so equality at range/aggression anchors matches XGBoost. An offline
code generator turns trusted trees into ordinary Python conditional
expressions with local feature variables. Generated source is imported once
at initialization and bound to the JSON model's SHA256; no per-decision code
generation, eval, or I/O occurs. The reference interpreter remains available
for parity testing and as an explicit performance fallback. Validation uses
10,000 actual hands, 10,000 randomized feature vectors, and threshold cases.
For the final model, all 21,968 generated/reference comparisons matched
exactly. The Linux compute measurement recorded about 93 microseconds mean
inference and no 1 ms overruns in 10,000 calls. A concurrent-load Windows
measurement recorded 166.2 microseconds median, 545.2 microseconds p99,
4.3304 milliseconds maximum, and 27/10,000 overruns. These are inference-only
measurements, not whole-agent latency guarantees. The root's complete decision
benchmarks and enforced-fallback tournament tests remain the relevant policy
timing evidence. Both platform reports are retained rather than discarding
the slower result.

No XGBoost or NumPy import occurs during gameplay. Missing
assets trigger deterministic approximations and are exposed by
`equity_model.asset_status()`; a complete candidate must include both assets.

`training/audit_equity.py` checks exact shared-board cases and measures the
policy's heads-up-to-multiway power approximation on a separate synthetic
calibration seed. Those results are tuning evidence, not final held-out
tournament evidence.

`training/calibrate_multiway.py` selects a small correction from 2,400 boards
(seed 1907719). Flop/turn use beta-distribution moments, conditioned on made
hand class and draws, to approximate correlation from future shared cards.
River uses a fitted exponent. On 1,800 untouched boards (seed 1907720) with
8,192 samples per opponent count, this lowered the HU-to-multiway RMSE from
0.05483 to 0.03501. The reported comparison starts from high-sample HU equity,
so it isolates this correction and does not include the learned HU model's
error. Opponents have identical ranges; unequal ranges and side pots remain
policy-level approximations. The fixed table and its provenance are included
in `equity_model.multiway_equity` and the JSON calibration measurement.

A subsequent ranking-based correction handles hero literally playing the
five-card board. Full kicker equality is checked, so such a hand can only tie
or lose. The approximation becomes `(2*q)**(n**1.05)/(n+1)` with `q<=.5`, since
heads-up share is half the tie probability. Opponent independence/card-removal
remains approximate; guaranteed ties at `q=.5` give the exact `1/(n+1)` share.
This correction was derived from hand-ranking semantics, separately from the
synthetic fit above. Regression tests cover every hand category and lowest
kicker improvements, including pair-board and five-flush-board cases.

XGBoost API reference used for the offline exporter:
https://xgboost.readthedocs.io/en/stable/python/python_api.html
