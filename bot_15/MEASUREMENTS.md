# Final measurements — bot_15

This is local experimentation, not contest submission or proof of official interface compatibility. The policy was frozen before the held-out scenarios below. No held-out result was used to change it.

## Frozen candidate

Selected `open1.2_pressure1.2` from tuning only: `{"adaptation": true, "aggression": 1.0, "bluff_scale": 1.0, "call_margin": 0.025, "icm_scale": 1.0, "open_scale": 1.2, "range_pressure": 1.2}`.

Initial sweep: 12,288 table runs; confirmation: 40,960. The predeclared selection objective was 80% average scenario score and 20% weakest-population average, normalized by equal-share points. All opponent pools and payout examples are synthetic. Tuning rotated absolute seats in lockstep with the starting button, so the initial relative position was fixed. This limitation was found before final reporting; final evaluation crosses all64seat/button pairs every64tables. The first incomplete evaluation was discarded, and the policy was not changed.

The selected policy's paired confirmation gain over the untuned policy was 0.08 [-0.02, 0.17] normalized units (95% normal interval). That interval includes zero: tuning does not establish a reliable improvement over the untuned policy.

## Held-out strategy evaluation

15 scenarios, 1,000 tables per scenario per policy, three policies: 45,000 table runs. Candidate seats rotate. Opponents, per-hand decks, and table seeds are paired across policies. Eliminations can cause later paths to diverge. Intervals are descriptive 95% normal intervals, without multiple-comparison correction.

All point vectors below are local sensitivity fixtures, scaled to total 100: winner-take-all; top3=50/30/20; flat3=34/33/33; linear=8:7:6:5:4:3:2:1 normalized to 100. Equal share is 12.5. Default blinds double every 25 hands; gentle blinds use 10/20,15/30,25/50,40/80,60/120,100/200,150/300,250/500. Neither full progression is claimed official.

| Field / points / blinds | Candidate points [95% CI] | Difference vs balanced baseline [95% CI] | Difference vs untuned bot [95% CI] |
|---|---:|---:|---:|
| mixed / winner / doubling | 34.00 [31.06, 36.94] | 19.60 [16.19, 23.01] | 4.70 [1.84, 7.56] |
| mixed / top3 / doubling | 24.90 [23.62, 26.18] | 10.66 [9.07, 12.25] | 0.53 [-0.56, 1.62] |
| tight / winner / doubling | 52.80 [49.70, 55.90] | 40.10 [36.54, 43.66] | 6.41 [2.90, 9.92] |
| tight / top3 / doubling | 28.88 [27.43, 30.33] | 18.79 [17.06, 20.52] | 2.03 [0.47, 3.59] |
| loose / winner / doubling | 42.70 [39.63, 45.77] | 11.10 [7.27, 14.93] | -0.90 [-4.06, 2.26] |
| loose / top3 / doubling | 35.29 [34.42, 36.17] | 8.40 [7.03, 9.77] | 0.30 [-0.53, 1.13] |
| aggressive / winner / doubling | 22.10 [19.53, 24.67] | 7.40 [4.10, 10.70] | -0.70 [-3.29, 1.89] |
| aggressive / top3 / doubling | 23.08 [21.91, 24.25] | 7.25 [5.73, 8.77] | 0.10 [-0.83, 1.03] |
| challenge / winner / doubling | 20.20 [17.71, 22.69] | 10.70 [7.70, 13.70] | 1.60 [-1.06, 4.26] |
| challenge / top3 / doubling | 15.21 [13.97, 16.45] | 6.36 [4.83, 7.89] | 0.64 [-0.65, 1.93] |
| mirror / winner / doubling | 12.20 [10.17, 14.23] | 8.80 [6.58, 11.02] | 0.90 [-1.65, 3.45] |
| mirror / top3 / doubling | 12.34 [11.25, 13.43] | 6.82 [5.53, 8.11] | -0.49 [-1.80, 0.82] |
| mixed / top3 / gentle | 24.45 [23.17, 25.72] | 10.05 [8.46, 11.65] | 0.15 [-0.95, 1.25] |
| mixed / flat3 / doubling | 22.31 [21.33, 23.29] | 9.57 [8.27, 10.86] | 0.35 [-0.51, 1.20] |
| mixed / linear / doubling | 17.27 [16.98, 17.55] | 3.84 [3.42, 4.26] | -0.01 [-0.27, 0.25] |

`challenge` consists of independently written equity-aware and polarized betting policies that were not used for tuning. They share the equity asset, so estimation errors are correlated. `mirror` uses seven copies of the untuned bot. Both are heuristics, not professional or equilibrium opponents. No average here predicts contest ranking.

## Full-decision latency and fallback behavior

Timing includes the complete in-process `Agent.act()` path with garbage collection enabled. Imports, asset loading/validation, JSON parsing, and I/O are excluded. Replay observations come from 200 separate tables; repeated replays are not independent strategic samples. OS scheduling and garbage collection remain part of wall time.

| Machine | Calls | Median us | p99 us | p99.9 us | Maximum us | Calls >1 ms | Startup ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linux compute node / Python 3.12.7 | 138,520 | 29.1 | 133.7 | 260.0 | 1902.3 | 1 | 813.2 |
| Windows desktop / Python 3.11.9 | 138,520 | 50.7 | 283.7 | 586.6 | 5435.4 | 29 | 1218.9 |

Internal error fallbacks during latency replays: Linux 0, Windows 0. First decision after reset: Linux max 226.6 us, Windows max 579.2 us. The Windows per-call CPU clock is too coarsely quantized for useful quantiles; wall time is the relevant reported measurement.

**The 1 ms bound is not guaranteed:** measured overruns are retained above, not discarded. The local engine implements the rule fallback (check if free, otherwise fold) for exceptions or over-budget calls. A collector must rerun timing with the actual adapter, hardware, and initialization policy.

Separate 1 ms-enforced stress evaluation: 3,000 candidate tables, 439,910 candidate decisions, 19 timed candidate fallbacks, 0 internal candidate error fallbacks, 0 table engine errors and 0 illegal actions. This uses a subset of held-out seeds and eight concurrent workers; it is not an independent extra strategy sample. Complete results are in `artifacts/heldout_enforced_subset.json`.

## Mechanics, model and offline packaging

- 69 unit/regression tests passed: betting order, heads-up/dead-button rules, full and short raises/reopening, pot eligibility, ties/odd chips, simultaneous eliminations, point/chip conservation, exact ICM reference comparisons, forced chops, blind-only all-ins, table reset, suit/seat relabeling and balanced benchmark seating.
- Exhaustive independent evaluator audit: all 2,598,960 five-card hands plus 10,000 seven-card subset comparisons; 19,664 native/Python feature comparisons including every quads turn board with all legal hole pairs. Numeric counts and scripts are retained.
- Final integration fuzz: 1,000 tables, 671,217 total actions, 137,942 candidate actions, 0 illegal actions, 0 engine errors, 0 internal candidate fallbacks. Unenforced wall overruns: 8.
- Equity model: 240,000 training hands, 12,000 validation, 12,000 held-out. The eight-tree runtime was selected on validation accuracy/latency. Held-out MAE 0.017264, RMSE 0.025904, 95th percentile absolute error 0.050961. Labels contain Monte Carlo noise. Asset size 5,160,408 bytes.
- The gameplay ZIP contains only whitelisted stdlib Python sources, numeric assets and documentation. The packaging command verifies hashes, extracts into a fresh directory, and runs import/action/reset checks under isolated Python with all socket audit events forbidden. See `artifacts/package_verification.json` for actual ZIP size and SHA-256.

## Limits and stopping rationale

No official starter interface, complete blind schedule or payout structure was available. No official compatibility or payout claims are made. The bot uses approximate opponent ranges, pot-layer multiway equity, draw/realization models and binary ICM risk scenarios. It does not solve the full game, account exactly for correlated multiway holdings or all future betting, or guarantee safety against adversarial strategies. Observation histories can omit actions after the bot folds, limiting opponent estimates. State must be reset at every table.

The larger 24-tree model improved validation MAE by only 0.00085 while substantially increasing latency. The confirmation sweep found modest and statistically uncertain gains among nearby policies. With independent mechanics checks, held-out stress scenarios, offline packaging and timing measurements complete, further local parameter search has diminishing evidentiary value without the actual contest interface and opponent population. No absolute optimality claim is made.

## Reproduce

```text
python -m unittest discover -s tests -v
python -m training.tune_policy --tables 128 --workers 8
python -m training.tune_policy --stage refinement --tables 512 --workers 8 --output artifacts/policy_refinement.json
python -m training.freeze_policy
python -m benchmarks.heldout --tables 1000 --workers 8
python -m benchmarks.heldout --tables 200 --workers 8 --enforce-1ms --output artifacts/heldout_enforced_subset.json
python -m benchmarks.fuzz --tables 1000 --seed 7410001 --output artifacts/integration_fuzz_final.json
python -m benchmarks.latency --tables 200 --repeat 5 --output artifacts/latency_local.json
python -m training.package_bot
```

All raw table rows, source hashes, tuning results and timing records remain in `artifacts/`. Remote work used only newly requested bounded bot_15 jobs in its authorized private directory; resource bookkeeping is in `.control/`. Nothing was pushed, published or submitted.
