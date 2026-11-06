# Offline equity training and inference

All strategy code and equity assets here were developed independently for this
bot. Gameplay uses the Python standard library only: no sampling, network,
subprocess, C++ executable, NumPy, or scikit-learn is needed to act.

The promoted model uses 84 semantic features and 320 gradient-boosted trees,
maximum depth 8, maximum 96 leaves per tree. It was fitted on 300,000 ordinary
random deals plus 100,000 independently generated hard cases. The selected model
JSON SHA-256 is:

`57fc9a7c9798b0a767011136d64a45297e60382f628fae35b982d58c5f98f0ad`

The prior runnable equity implementation is preserved in
`training/checkpoint_v2/`. The root README documents the final policy, packaging,
full-agent timing, tournament benchmarks, and final held-out evaluation.

## What the targets mean

Cards are `rank*4+suit`, with rank zero representing a deuce. Both the original
C++ generator and Python runtime evaluate the exact best five-card Hold'em hand,
including wheel straights and split pots. A 10,000-hand random cross-language
score comparison passed during development. Additional independent mechanics
checks live in the simulator and unit tests.

The generator estimates heads-up random-hand equity for all 169 canonical
starting hands, then ranks the 1,326 physical combinations using those estimates.
It constructs nominal top 10%, 25%, 50%, and 100% ranges. Whole hand types are
retained at the boundaries, giving 134, 334, 670, and 1,326 combinations with the
saved ranking. This is an empirical equity ordering, not an equilibrium opening
chart or a claim about actual opponents.

Each postflop row contains a random hero hand and flop, turn, or river;
1, 2, 3, 5, or 7 opponents; a range fraction; and a synthetic betting-pressure
level of 0, 1, or 2. Hero cards, known board cards, opponent cards, and runouts are
sampled without overlap. The target is expected pot share, including
`1 / tied_winners` for ties. Multiway targets use shared runouts and are simulated
directly. Heads-up river targets enumerate every legal range combination.

Pressure 0 leaves the preflop range uniform. Pressure 1 applies explicit heuristic
weights favoring private pairs, stronger made hands, and draws. Pressure 2 cubes
those weights with a small air floor. The exact function is `weight()` in
`generate.cpp`. The policy chooses pressure from public betting history. These
priors are approximate: they do not solve the betting game, and the initial
straight-draw prior sometimes retains board-only draws. Training observes only
discrete range/pressure settings, so predictions between them are piecewise.

Opponent hands are sampled sequentially, excluding cards already selected.
This captures blockers and shared-board correlation, but is not the exact joint
posterior from independently sampling all ranges and then conditioning the entire
assignment on no collisions. The approximation matters most with many narrow
opponents. Failed assignments restart after bounded candidate retries.

## Premium preflop tables

Pure random-hand equity ranks medium pairs above some premium unpaired hands.
For large reraises, separate explicit priors are used:

- Nominal 2.5%: QQ+, AK; 34 combinations, or 2.56%.
- Nominal 5%: TT+, AQ+, AJs, KQs; 70 combinations, or 5.28%.

`narrow.cpp` estimates every canonical hero hand against these priors for one,
two, and three opponents, with 25,000 samples per entry. Runtime interpolates
between the saved preflop tables. For four through seven opponents in the narrow
priors it uses the bounded extrapolation `e3 * (e3/e2)^(n-3)`; those rare cases
are not calibrated. The original 10% through 100% tables simulate every opponent
count from one through seven. Postflop models clamp ranges below 10%.

An independent 25,000-sample comparison across the 1,014 narrow entries had mean
absolute difference 0.00290, RMSE difference 0.00358, and maximum difference
0.01284. These compare two noisy estimates, not estimates against exact truth.
See `benchmarks/narrow_validation.json` for the recorded values and seeds.

## Features, exact bounds, and fast inference

The original 58 features describe made hands, private and board ranks, suit
texture, draws, blockers, opponent count, range fraction, and pressure. Version 3
adds 26 features describing improvement over the board and the maximum legal
opposing straight, flush, full house, quads, trips, and straight flush with hero
blockers. These help the regressor distinguish cases such as a private 3 upgrading
a `22233` board to `33322`. All features are invariant under suit permutation and
hole/board ordering.

The maximum opposing category and relevant ranks matched exhaustive enumeration
on 150 random flop, turn, and river boards. Unit tests repeat an independent
sample. `bot/exact.py` also certifies some strictly winning river hands: every
possible opponent hand must lose. The guarantee was checked on 120 independently
sampled river hands against all legal opponent combinations. Shared nuts are
never treated as certain wins. General certificates apply only on the river;
royal flushes have a separate exact override on earlier streets as well.

When the hero plays the five-card board, expected pot share is bounded above by
`1/(opponents+1)`, because every opponent can at least tie. This bound is applied
to the regression output. The policy separately handles guaranteed board chops.

`compile_model.py` converts numeric JSON trees into nested Python conditional
expressions over unpacked feature locals. Integral features use floored integer
cutpoints, preserving branch decisions while enabling fast integer comparisons.
The final range-fraction and pressure features retain floating cutpoints. The
compiled module embeds the source JSON SHA-256 and feature version. Runtime
checks both at import and retains a reference interpreter if they do not match.
Compiled and interpreted outputs matched exactly on all 8,000 validation rows.

## Evidence and limits

For the selected runtime, the 8,000 model-selection validation rows yielded:

| Metric | Value |
|---|---:|
| Mean absolute error | 0.02085 |
| Root mean squared error | 0.03202 |
| 99th percentile absolute error | 0.10907 |
| Maximum absolute error | 0.42939 |

See `benchmarks/equity_validation_v3_final.json` for calibration, slices, worst
cases, model hash, environment, and timing outliers. These are validation results,
not an untouched test result or contest performance. The largest remaining tail
is a top-set case against many restricted opponents whose range excludes its
only beating straights. Rare river hands and restricted multiway ranges remain
meaningful approximation risks.

An independent Python audit recomputed 400 exact river targets across all three
pressure levels, matching within 4.98e-9. Eighteen flop/turn cases were simulated
independently with 6,000 samples each; the RMSE difference from their 384-sample
C++ labels was 0.01550. See `benchmarks/generator_audit.json`.

One isolated local timing run measured feature extraction plus compiled inference
at median 114.9 us and p99 255.2 us, with no calls over 1 ms in 8,000 calls.
A separate full `postflop` run observed 15 calls over 1 ms among 16,000 calls,
including a 38.3 ms scheduling outlier. These inference-only development timings
are superseded by final full-agent benchmarks. No official timing guarantee is
claimed; the official adapter and measurement boundary are unavailable.

## Reproduction

Use a bounded private compute job for substantial work. Following the user's
later instruction, do not run bulk training, simulations, builds, or benchmarks
on the laptop. The head agent maintains the authorized job ledger.

Training used g++ 12.2, NumPy 1.26.4, and scikit-learn 1.5.1 on Linux. Those
packages are needed only for fitting. Runtime requires Python 3.10 or later and
was tested on Python 3.11. The saved assets avoid training dependencies entirely.
Run the following from the workspace root in an authorized compute allocation:

```sh
export OMP_NUM_THREADS=8
g++ -O3 -fopenmp training/generate.cpp -o training/generate
g++ -O3 -fopenmp training/generate_hard.cpp -o training/generate_hard
g++ -O3 -fopenmp training/narrow.cpp -o training/narrow

training/generate dataset training/train.csv 300000 384 161001 assets/strength.txt
training/generate dataset training/validation.csv 8000 2048 161002 assets/strength.txt
training/generate dataset training/test.csv 8000 2048 161003 assets/strength.txt
training/generate preflop training/preflop.csv 16000 30000 161004 assets/strength.txt
training/generate_hard training/hard.csv 100000 768 161011 assets/strength.txt
training/narrow training/preflop_narrow.csv 25000 161005
python training/export_narrow.py

python training/train.py --extra-train training/hard.csv --features-module bot.features --iterations 320 --depth 8 --leaves 96 --output assets/equity_model.json --metrics training/model_reproduction_metrics.json
python training/compile_model.py
```

Keep `assets/strength.txt` for exact range-order reproducibility. It was originally
generated by the smoke command below; missing strength files are regenerated with
30,000 samples per canonical hand and a seed offset of 999:

```sh
training/generate dataset training/smoke.csv 1000 128 161100 assets/strength.txt
```

Model fitting and hyperparameter selection used only training and validation
seeds. `train.py` does not read test targets unless explicitly given `--test`.
After model and policy freeze, evaluate the separate test dataset with:

```sh
python training/evaluate.py --test training/test.csv --split held_out --output benchmarks/equity_test.json --latency-repeats 2
python -m unittest tests.test_equity tests.test_features_v3 tests.test_exact_wins -v
```

To reproduce the generator audit and independent narrow-table comparison:

```sh
training/generate dataset training/review.csv 10000 384 161111 assets/strength.txt
python training/verify_generator.py
training/narrow training/preflop_narrow_validation.csv 25000 161006
```

`training/features_base_v2.py` and `training/features_v3.py` retain the staged
feature definitions used for model selection. The promoted runtime equivalents
are `bot/features_base.py` and `bot/features.py`. Reproduction commands use the
promoted runtime features, which preserve the staged feature values exactly.
