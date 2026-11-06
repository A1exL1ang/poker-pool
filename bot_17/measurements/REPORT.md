# bot_17 measurement report

The base policy was frozen before reserved tournament evaluation. The full
local interface and assumptions are in `CONTRACT.md` and `README.md`. This
report concerns an independently authored simulator and synthetic opponents;
it is not an official contest result or a claim of optimal play.

Final held-out evaluation completed **8,500 tables and 1,201,464 hero decisions**
with zero exceptions, illegal actions or internal error fallbacks. There were
**four wall-clock timeout fallbacks**, all enforced as check/fold. Selected
source hashes match every evaluation run. The strategy was not changed after
these results were examined. See `validation_summary.json`.

## Development and selection

The main strategy sweep used 9,000 tables: ten profiles, six environments,
150 tables per cell, starting at seed 210000 with disjoint environment blocks.
Confirmation used another 9,000 tables: base, cautious and tighter-call
profiles, six environments, 500 tables per cell, starting at seed 510000.
Both phases were for selection, not held-out evaluation.

| Profile | Sweep mean | Confirmation mean | Confirmation difference from base, descriptive 95% interval |
|---|---:|---:|---:|
| Base | 0.30870 | 0.31079 | Reference |
| Cautious | 0.31214 | 0.31188 | 0.00109 [-0.00786, 0.01004] |
| Tighter calls | 0.31151 | 0.31608 | 0.00528 [-0.00385, 0.01442] |

Means average six explicitly chosen synthetic environments; all six payout
vectors sum to one. The intervals are paired by seed, descriptive, and not
adjusted for searching profiles. No alternative established a clear gain.
Tighter calls lost 0.009 points against baseline-policy opponents in
confirmation selfplay, so the base policy was retained. Selection and source
hashes were recorded in `selection.json` before reserved seeds were run.
`tune/` and `confirm/` retain per-table outcomes and comparisons.

## Held-out tournament evaluation

All rows below enforce the 1 ms deadline. `Top3` means the illustrative vector
[0.5, 0.3, 0.2, 0, 0, 0, 0, 0], and winner-only means [1, 0, …]. These are
explicit experimental payouts, not official points. Their equal-share
reference is 0.125; that reference is not a measured competing strategy.

| Field / experimental payout | Tables | Mean points | Approximate 95% interval |
|---|---:|---:|---:|
| Mixed / top3 | 1,000 | 0.29045 | [0.27723, 0.30367] |
| Mixed / winner-only, fast blinds | 1,000 | 0.46000 | [0.42909, 0.49091] |
| Tight / top3 | 1,000 | 0.29745 | [0.28357, 0.31133] |
| Pressure / top3 | 1,000 | 0.26095 | [0.24901, 0.27289] |
| Calling / top3, slow blinds | 1,000 | 0.43964 | [0.42968, 0.44961] |
| Identical-policy selfplay / top3, fast blinds | 1,000 | 0.12860 | [0.11740, 0.13980] |
| Previously unused all-shove field / top3 | 500 | 0.28512 | [0.26886, 0.30138] |
| Previously unused all-random field / top3 | 500 | 0.42344 | [0.41036, 0.43652] |
| Mixed / linear [8,7,6,5,4,3,2,1] | 500 | 6.64088 | [6.50257, 6.77918] |
| Mixed / no payout input, terminal chip share | 500 | 0.48289 | [0.44040, 0.52537] |
| Python fallback, mixed / top3 | 250 | 0.29837 | [0.27218, 0.32455] |
| Python fallback selfplay / top3 | 250 | 0.12784 | [0.10596, 0.14972] |

Linear points and chip share are different units from the other rows and
must not be averaged into them. Selfplay intervals contain the symmetry value
0.125. Baseline opponents are simple, locally authored stress tests; beating
them does not establish strength against unknown contest entrants.

The first six rows use seeds starting at 910000, 930000, 950000, 970000,
990000 and 1010000 in the environment order defined in `tools/experiment.py`.
Extra rows start at 1100000 (shove), 1120000 (random), 1140000 (linear),
1160000 (chip share), 1200000 (Python mixed) and 1220000 (Python selfplay).
All are separate from policy training/confirmation. Full outcomes and
settings are in `heldout/` and `heldout_extra/`.

The four enforced overruns occurred once each in mixed-top3, pressure-top3,
calling-top3 and linear-payout evaluation. Largest observed game decision was
17.417 ms. There were 1,126,803 native decisions and 74,661 Python-fallback
decisions; none of the latter exceeded the deadline in these remote games.
Across all eight seats, there were zero errors or invalid actions and seven
timeout fallbacks, including the four hero fallbacks.
The 6,000-table primary set ran with six workers, followed by four-worker
extra scenarios and two-worker Python checks within one eight-CPU allocation.

Reproduce the main evaluation with the README command. Extra scenarios use
`python tournament.py --tables N --seed S --lineup FIELD --payouts PAYOUT
--enforce-time --output FILE`, with the counts/seeds/fields above. Set
`POKER_PURE_PYTHON=1` before launching the last two rows. Native default blind
levels apply to the four extra native fields and both extra Python fields.

## Mechanics and offline checks

- 73 tests pass in final Linux native execution; the earlier Windows suite
  also passed. Pure Python has two
  native-only tests skipped; the other tests exercise the same strategy,
  simulator, side-pot, ICM, action and fallback semantics.
- The native hand evaluator was checked over every 5-card combination
  (2,598,960), plus 5,400 comparisons to an independent five-card-subset oracle.
- Simulator tests include 3,000 randomized legal-action hands, conservation,
  heads-up and short blinds, full and cumulative short raises, per-player
  reopening, uncalled refunds, side pots, ties/odd chips, dead buttons and ICM.
- ICM was independently cross-checked on 500 randomized configurations and
  100 capped heads-up pot-odds cases. Strategy regressions cover guaranteed
  split boards, dead-small-blind positions, and a tiny all-in excluded from a
  large side pot.
- Gameplay file/network APIs are blocked in a test after initialization.
  Archive hashes and native/Python imports are also checked in isolated
  processes from the extracted runtime package.

## Equity assets and estimator evaluation

Preflop: 169 classes × 7 range widths × 7 opponent counts × 20,000 trials =
165,620,000 generated trials. No official data or other team's code was used.

Python surrogate: 100,000 training states, 256 teacher samples per state;
6,000 validation states, 2,048 samples each. The selected model has 160
depth-six boosted trees; four or more distinct side pots use the first 80
trees to bound inference work. Models and static inference code ship together.

| Frozen estimator audit | States | Python RMSE / MAE | Native RMSE / MAE |
|---|---:|---:|---:|
| Ordinary equity | 4,000 | 0.04918 / 0.03103 | 0.03421 / 0.02219 |
| Heterogeneous seven-pot equity | 600 | 0.04471 / 0.02653 | 0.02997 / 0.02088 |

Reference equity used 4,096 samples, or exact modeled heads-up river
enumeration. These are disjoint seeds after model selection. The Python model
has large rare errors: max absolute error 0.54666 in ordinary states and
0.56958 in side-pot states. Native is preferred. Both share heuristic opponent
range assumptions; a low model error does not validate those assumptions.
The side-pot teacher/runtime audit recorded four proposal-cap fallbacks and
one uniform emergency fallback across 2,496,000 trials. Full records and
seeds are in `../data/equity_model_heldout.json` and
`../data/equity_sidepots_heldout.json`.

## Local decision timing

50,000 warm whole-`Agent.act` decisions per run, with normal garbage
collection enabled and real-game plus ICM/multiway/seven-pot stress states.
Models are loaded before timing. Pure and native runs generate slightly
different natural-play corpora; these are not paired latency comparisons.

| Environment/backend | Median µs | p99 µs | p99.9 µs | Maximum µs | >1 ms |
|---|---:|---:|---:|---:|---:|
| Windows Python 3.11, native | 89.8 | 339.7 | 658.5 | 3,397.0 | 14 / 50,000 |
| Windows Python 3.11, Python model | 112.5 | 473.5 | 793.0 | 3,333.5 | 15 / 50,000 |
| WSL Python 3.12, native | 130.9 | 651.8 | 2,647.2 | 9,982.8 | 249 / 50,000 |
| Engaging Linux Python 3.12, native | 67.4 | 189.0 | 235.5 | 627.8 | 0 / 50,000 |
| Engaging Linux Python 3.12, Python model | 56.2 | 235.9 | 265.2 | 588.3 | 0 / 50,000 |

Each run recorded zero exceptions, illegal actions and internal error
fallbacks. Native runs recorded zero range-proposal-cap/emergency fallbacks in
these decision corpora. Wall-clock overruns remain real and cannot be dismissed
as guaranteed harmless scheduling effects. On WSL, 234 calls also exceeded
1 ms of measured thread CPU time. Windows thread CPU accounting was too
coarsely quantized to interpret per call. Initialization/import took roughly
152 ms on Windows native, 400 ms in WSL and 375 ms on Engaging native, and is
excluded from warm timing. Engaging constructor time was about 14 µs.
If an official timer includes initialization or transport, the present local
timing contract is insufficient.

Earlier measurements remain in the workspace, including more contended runs;
they have not been removed to improve the presentation. Remote serial timing
ran after all parallel evaluation workers had exited. Its clean 50,000-call
result does not supersede the real overruns recorded in games or locally.

The complete rules/mechanics tests passed remotely: 73 native tests, and 73
Python-mode tests with two native-only tests skipped. Four owned bounded
Engaging jobs completed successfully: preflop generation (10 s), tuning
(87 s), confirmation (103 s), and final validation (177 s). Each used at most
eight CPUs and 12 GB, with no overlapping team allocations or use of existing
user allocations. All owned jobs are complete; the shared SSH master remains
untouched.

## Remaining integration limits

The official Agent interface, noninitial blinds, actual place points, sandbox
ABI and precise timer boundary were unavailable. The native library is x64
and may require rebuilding; Python fallback works without it but is less
accurate. ICM call utility assumes a binary heads-up showdown and is not a full
tournament action-value solver. Multiway ranges, future betting, and
fold-equity estimates remain approximate. Unknown opponents may exploit this
policy. The simulator's finishing-tie rule and no-ante assumption require
official confirmation.

No bot has been submitted, published or pushed. Operational metadata and the
team's own bounded compute job ledger are in `.control`, outside the runtime
archive. Both helper agents finished. On reading the user's sleep-time update,
no local heavy task was running; remaining substantial work was kept on the
team's own Engaging allocation.
