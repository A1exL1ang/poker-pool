# Measured results

These are independent local experiments, not contest scores. The official interface,
payouts and later blind schedule were unavailable. All distributions below are
explicit hypothetical sensitivity cases. No submission, push or publication occurred.

## Locked held-out tournament evaluation

Selected parameters were frozen after 28 candidates on tune seeds and independent validation;
validation paired gain was 0.01684 normalized points (SE 0.00839).
Final evaluation never selects or modifies the policy. It uses fresh seeds, six
opponent pools, five payout cases and three blind schedules. The diverse opponent
pool, top-two/satellite payouts and slow schedule were absent from policy tuning.
All policies and models are original local implementations. The same final model
assets are shared by the applicable reference opponents.

| Policy | Tables | Mean normalized points | 95% normal CI |
|---|---:|---:|---|
| selected | 5,400 | 0.19310 | [0.18665, 0.19955] |
| reference | 5,400 | 0.19588 | [0.18952, 0.20224] |
| static_tag | 1,800 | 0.13407 | [0.12504, 0.14309] |

Selected minus reference on common deck seeds: **-0.00278**
(paired 95% CI [-0.00926, +0.00370]).
The interval does not establish a positive improvement over the reference.

Payout totals are normalized to one; an exchangeable seat averages 0.125.
Averaging across these hypothetical schedules is a local tuning objective.
The intervals describe sampling uncertainty, not generalization to contest bots.

| Selected policy opponent pool | Tables | Mean points | 95% CI |
|---|---:|---:|---|
| mixed | 900 | 0.21250 | [0.19615, 0.22884] |
| tight | 900 | 0.20791 | [0.19173, 0.22410] |
| pressure | 900 | 0.21177 | [0.19548, 0.22805] |
| calling | 900 | 0.26664 | [0.24928, 0.28400] |
| mirror | 900 | 0.12395 | [0.11162, 0.13628] |
| diverse | 900 | 0.13584 | [0.12212, 0.14956] |

Selected policy: 908,965 decisions, 0 illegal actions,
0 raised exceptions, 0 internal error fallbacks. The normal
parallel tournament run recorded 5 wall-clock overruns without enforcing them.

## Decision timing and fallbacks

Final timing platform: `Linux-4.18.0-553.147.1.el8_10.x86_64-x86_64-with-glibc2.28`, Python 3.12.7.
The boundary is in-process `Agent.act`, excluding imports, model loading,
reset, observation construction and JSON/process IO. GC remains enabled.
These finite measurements are not an absolute deadline guarantee.

| Measurement | Calls | Mean µs | p99 µs | Maximum µs | >1 ms |
|---|---:|---:|---:|---:|---:|
| Real gameplay trace | 41,458 | 58.1 | 170.2 | 1716.2 | 2 |
| Cold-history replay | 124,374 | 59.5 | 174.6 | 693.1 | 0 |
| eight_unequal_unopened | 3,000 | 469.6 | 514.1 | 669.3 | 0 |
| eight_unequal_limped_bb | 3,000 | 481.8 | 524.6 | 979.3 | 0 |
| eight_equal_unopened | 3,000 | 368.0 | 410.9 | 558.1 | 0 |
| eight_with_forced_allin_small_blind | 3,000 | 363.2 | 403.0 | 548.0 | 0 |
| eight_live_seats_isolated_icm_top3 | 3,000 | 253.9 | 295.3 | 430.8 | 0 |
| eight_live_seats_isolated_icm_linear | 3,000 | 343.4 | 385.6 | 516.4 | 0 |
| eight_live_seats_isolated_icm_satellite | 3,000 | 217.5 | 230.8 | 392.9 | 0 |

Separate enforced-deadline run: 1,200 tables, 204,444 hero decisions,
**11 timeout check/fold replacements**, 0 exceptions,
0 illegal actions and 0 internal fallbacks.
Mean normalized points in that different-seed run: 0.18876.
Timeout-enforced outcomes depend on OS scheduling and are therefore not exactly reproducible.
Earlier concurrent Windows runs also exceeded 1 ms; their original JSON reports
are retained. Windows per-call thread CPU readings were quantized to roughly
15.625 ms and cannot identify the cause of individual stalls.

## Equity accuracy and mechanics

The final model was selected using validation only. A fresh 30,000-state set
(seed 1900205, 8,192 simulation draws per applicable state) was then evaluated.
Errors below measure the declared synthetic range distribution, not strategy quality.

| Street | Test states | MAE | RMSE | 99th-percentile absolute error |
|---|---:|---:|---:|---:|
| Flop | 10,100 | 0.01155 | 0.01623 | 0.05281 |
| Turn | 9,952 | 0.01532 | 0.02276 | 0.07799 |
| River | 9,948 | 0.01831 | 0.03159 | 0.11548 |

- 61 final unit tests passed, including independent side-pot/ICM checks,
  minimum raises/reopening, short all-in blinds, heads-up/dead-button rules,
  state reset, card secrecy, public observations and offline decision calls.
- Exhaustive 2,598,960 five-card rank census plus independent native/Python
  evaluator and feature comparisons passed.
- Generated inference: 41,968
  comparisons, maximum difference 0.0 against the reference.
- A separately calibrated multiway correction reduced isolated test RMSE
  from 0.05483 to 0.03501. Board-playing tie handling has separate tests.
- Large individual equity errors remain; see maximum and tail errors in JSON.

## Reproduction and limits

See `training/final_job.sh` for the exact final command sequence; full per-table
rows, seeds, source/model hashes and sensitivity breakdowns are in `measurements/`.
`training/tune_job.sh`, `training/refine_job.sh` and `training/job.sh` preserve
the training/selection commands. `build_package.py` checks model assets, creates
a deterministic offline ZIP and smoke-tests its fresh extraction using isolated
Python with networking blocked and decision-time file access blocked.

The strategy remains a compact range model and heuristic action policy, with
approximated opponent behavior and future realization. It has no equilibrium,
exploitability, official compatibility, contest-win or absolute timing guarantee.
