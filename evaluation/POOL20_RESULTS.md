# Twenty-bot evaluation

All 20 frozen bots were evaluated on MIT Engaging in 82,080 eight-seat tournaments. The complete schedule, chip/prize conservation, and fault checks passed. No bot policy was changed after collection.

## Main findings

- Bot 18 has the highest point estimate under top-three and top-four payouts; bot 6 leads winner-take-all. Top-three bot 18 and bot 6 means differ by only 0.016 percentage points and have strongly overlapping intervals. Do not treat this as a decisive winner.
- PPO Claude exceeds every bot in same-table comparisons under all three payout shapes (each marginal 95% interval favors PPO). Pure-net Claude also generally leads; bot 4/top3 and bot 16/top4 comparisons are inconclusive.
- PPO beats pure-net Claude directly at the same tables by 4.04 percentage points for top3 (95% CI 3.62â€“4.47), 3.78 for winner (2.91â€“4.66), and 3.73 for top4 (3.41â€“4.06).

## All-20-only rankings

Scores are mean normalized payout points per table, expressed as percentages of one total prize pool. Equal-share reference:12.5%. Intervals are marginal field-seed-cluster bootstrap intervals; rank order is by point estimate, not proven superiority.

### top3

| Rank | Bot | Mean points (%) | 95% CI (%) | Bootstrap rank interval |
|---:|---:|---:|---|---|
| 1 | 18 | 13.742 | 13.196â€“14.283 | 1â€“7 |
| 2 | 6 | 13.725 | 13.178â€“14.288 | 1â€“7 |
| 3 | 4 | 13.583 | 12.992â€“14.183 | 1â€“8 |
| 4 | 8 | 13.399 | 12.824â€“13.979 | 1â€“9 |
| 5 | 9 | 13.372 | 12.781â€“13.960 | 1â€“10 |
| 6 | 14 | 13.289 | 12.729â€“13.856 | 1â€“10 |
| 7 | 11 | 13.229 | 12.624â€“13.805 | 2â€“11 |
| 8 | 16 | 13.059 | 12.489â€“13.621 | 3â€“12 |
| 9 | 2 | 12.741 | 12.209â€“13.302 | 5â€“13 |
| 10 | 13 | 12.738 | 12.199â€“13.284 | 6â€“13 |
| 11 | 5 | 12.662 | 12.142â€“13.197 | 6â€“14 |
| 12 | 20 | 12.497 | 11.960â€“13.033 | 8â€“14 |
| 13 | 15 | 12.316 | 11.747â€“12.890 | 9â€“15 |
| 14 | 7 | 12.053 | 11.504â€“12.601 | 11â€“17 |
| 15 | 12 | 11.705 | 11.166â€“12.255 | 13â€“18 |
| 16 | 19 | 11.697 | 11.144â€“12.270 | 13â€“18 |
| 17 | 10 | 11.549 | 11.017â€“12.050 | 14â€“19 |
| 18 | 3 | 11.086 | 10.547â€“11.621 | 16â€“20 |
| 19 | 1 | 11.020 | 10.471â€“11.556 | 16â€“20 |
| 20 | 17 | 10.537 | 10.031â€“11.073 | 18â€“20 |

### winner

| Rank | Bot | Mean points (%) | 95% CI (%) | Bootstrap rank interval |
|---:|---:|---:|---|---|
| 1 | 6 | 16.612 | 15.420â€“17.808 | 1â€“2 |
| 2 | 11 | 15.351 | 14.117â€“16.578 | 1â€“7 |
| 3 | 8 | 14.583 | 13.490â€“15.705 | 2â€“9 |
| 4 | 13 | 14.446 | 13.298â€“15.604 | 2â€“9 |
| 5 | 20 | 14.391 | 13.286â€“15.534 | 2â€“9 |
| 6 | 4 | 14.337 | 13.201â€“15.465 | 2â€“9 |
| 7 | 9 | 14.227 | 13.147â€“15.309 | 2â€“10 |
| 8 | 14 | 14.172 | 13.074â€“15.249 | 2â€“10 |
| 9 | 15 | 13.268 | 12.146â€“14.387 | 5â€“11 |
| 10 | 2 | 13.185 | 12.169â€“14.236 | 6â€“10 |
| 11 | 7 | 11.458 | 10.412â€“12.526 | 11â€“16 |
| 12 | 18 | 11.349 | 10.327â€“12.360 | 11â€“16 |
| 13 | 1 | 11.321 | 10.303â€“12.315 | 11â€“16 |
| 14 | 19 | 11.102 | 10.064â€“12.156 | 11â€“17 |
| 15 | 16 | 10.855 | 9.904â€“11.777 | 11â€“18 |
| 16 | 3 | 10.663 | 9.710â€“11.607 | 12â€“18 |
| 17 | 17 | 9.868 | 8.908â€“10.852 | 15â€“20 |
| 18 | 12 | 9.841 | 8.859â€“10.824 | 15â€“20 |
| 19 | 5 | 9.759 | 8.807â€“10.698 | 15â€“20 |
| 20 | 10 | 9.211 | 8.258â€“10.156 | 17â€“20 |

### top4

| Rank | Bot | Mean points (%) | 95% CI (%) | Bootstrap rank interval |
|---:|---:|---:|---|---|
| 1 | 18 | 13.780 | 13.353â€“14.203 | 1â€“4 |
| 2 | 6 | 13.522 | 13.034â€“13.994 | 1â€“6 |
| 3 | 16 | 13.465 | 13.018â€“13.913 | 1â€“6 |
| 4 | 8 | 13.309 | 12.832â€“13.808 | 1â€“8 |
| 5 | 2 | 13.010 | 12.549â€“13.478 | 3â€“11 |
| 6 | 14 | 12.993 | 12.542â€“13.453 | 3â€“12 |
| 7 | 9 | 12.840 | 12.395â€“13.304 | 4â€“13 |
| 8 | 4 | 12.815 | 12.349â€“13.279 | 4â€“13 |
| 9 | 20 | 12.793 | 12.390â€“13.198 | 5â€“13 |
| 10 | 11 | 12.703 | 12.224â€“13.188 | 4â€“14 |
| 11 | 5 | 12.645 | 12.187â€“13.096 | 5â€“15 |
| 12 | 15 | 12.441 | 11.969â€“12.910 | 7â€“16 |
| 13 | 13 | 12.423 | 11.967â€“12.905 | 7â€“16 |
| 14 | 7 | 12.368 | 11.920â€“12.808 | 8â€“16 |
| 15 | 12 | 12.231 | 11.792â€“12.668 | 10â€“17 |
| 16 | 10 | 11.791 | 11.332â€“12.237 | 14â€“18 |
| 17 | 19 | 11.768 | 11.295â€“12.234 | 14â€“18 |
| 18 | 3 | 11.118 | 10.687â€“11.555 | 17â€“20 |
| 19 | 1 | 11.050 | 10.599â€“11.502 | 18â€“20 |
| 20 | 17 | 10.932 | 10.487â€“11.370 | 18â€“20 |

## Matched baseline comparisons

Each difference is bot minus baseline in the SAME mixed-field tables. Each mixed field contains both Claude variants and six of our bots. A negative difference favors Claude; these are multiplayer comparisons, not heads-up matches.

### top3

| Bot | Versus PPO (pp; 95% CI) | Versus pure net (pp; 95% CI) |
|---:|---|---|
| 1 | -7.515 [-8.262, -6.740] | -3.971 [-4.654, -3.254] |
| 2 | -7.153 [-7.952, -6.349] | -2.758 [-3.471, -2.026] |
| 3 | -6.113 [-6.830, -5.379] | -2.029 [-2.708, -1.307] |
| 4 | -4.879 [-5.666, -4.090] | -0.680 [-1.382, +0.053] |
| 5 | -6.904 [-7.640, -6.172] | -3.238 [-3.921, -2.551] |
| 6 | -5.371 [-6.147, -4.623] | -1.383 [-2.091, -0.684] |
| 7 | -6.552 [-7.262, -5.837] | -2.533 [-3.235, -1.836] |
| 8 | -4.762 [-5.524, -4.033] | -1.255 [-1.978, -0.545] |
| 9 | -6.610 [-7.379, -5.813] | -2.473 [-3.161, -1.725] |
| 10 | -7.133 [-7.855, -6.408] | -2.973 [-3.643, -2.316] |
| 11 | -5.671 [-6.491, -4.847] | -1.886 [-2.619, -1.168] |
| 12 | -7.887 [-8.592, -7.166] | -4.048 [-4.722, -3.370] |
| 13 | -5.596 [-6.400, -4.789] | -1.826 [-2.536, -1.115] |
| 14 | -5.291 [-6.068, -4.520] | -1.303 [-1.968, -0.613] |
| 15 | -6.760 [-7.566, -5.950] | -3.403 [-4.124, -2.660] |
| 16 | -5.413 [-6.193, -4.650] | -0.773 [-1.487, -0.061] |
| 17 | -8.604 [-9.334, -7.876] | -3.891 [-4.552, -3.228] |
| 18 | -6.157 [-6.946, -5.393] | -1.859 [-2.561, -1.154] |
| 19 | -6.263 [-7.045, -5.507] | -2.160 [-2.855, -1.466] |
| 20 | -6.155 [-6.930, -5.387] | -1.579 [-2.252, -0.893] |

### winner

| Bot | Versus PPO (pp; 95% CI) | Versus pure net (pp; 95% CI) |
|---:|---|---|
| 1 | -9.375 [-10.806, -7.954] | -5.556 [-6.857, -4.221] |
| 2 | -10.069 [-11.496, -8.576] | -5.683 [-6.992, -4.256] |
| 3 | -8.790 [-10.189, -7.313] | -6.031 [-7.437, -4.608] |
| 4 | -8.004 [-9.488, -6.519] | -4.532 [-5.953, -3.071] |
| 5 | -12.043 [-13.462, -10.612] | -8.516 [-9.842, -7.164] |
| 6 | -6.689 [-8.176, -5.133] | -3.600 [-5.070, -2.080] |
| 7 | -10.490 [-11.927, -9.042] | -6.780 [-8.184, -5.362] |
| 8 | -6.853 [-8.284, -5.423] | -2.650 [-4.067, -1.270] |
| 9 | -9.905 [-11.369, -8.411] | -6.561 [-8.010, -5.148] |
| 10 | -11.349 [-12.778, -9.975] | -7.730 [-9.108, -6.397] |
| 11 | -8.991 [-10.490, -7.518] | -4.423 [-5.837, -3.050] |
| 12 | -12.299 [-13.671, -10.896] | -8.516 [-9.745, -7.281] |
| 13 | -6.597 [-8.095, -5.106] | -2.814 [-4.228, -1.417] |
| 14 | -7.420 [-8.887, -5.892] | -4.368 [-5.745, -2.960] |
| 15 | -9.722 [-11.135, -8.314] | -5.629 [-7.021, -4.316] |
| 16 | -9.521 [-10.945, -8.102] | -6.323 [-7.711, -4.974] |
| 17 | -12.409 [-13.779, -11.006] | -7.913 [-9.195, -6.532] |
| 18 | -12.445 [-13.837, -11.045] | -7.456 [-8.759, -6.105] |
| 19 | -9.850 [-11.289, -8.418] | -6.049 [-7.405, -4.735] |
| 20 | -8.077 [-9.558, -6.582] | -4.112 [-5.583, -2.713] |

### top4

| Bot | Versus PPO (pp; 95% CI) | Versus pure net (pp; 95% CI) |
|---:|---|---|
| 1 | -7.036 [-7.640, -6.416] | -3.295 [-3.846, -2.743] |
| 2 | -6.018 [-6.628, -5.418] | -2.204 [-2.770, -1.623] |
| 3 | -5.808 [-6.422, -5.185] | -2.730 [-3.309, -2.156] |
| 4 | -4.406 [-5.011, -3.803] | -1.093 [-1.667, -0.514] |
| 5 | -6.371 [-6.950, -5.794] | -2.476 [-3.016, -1.923] |
| 6 | -5.404 [-6.015, -4.752] | -0.775 [-1.377, -0.191] |
| 7 | -5.707 [-6.368, -5.076] | -2.050 [-2.628, -1.495] |
| 8 | -4.170 [-4.820, -3.513] | -0.945 [-1.574, -0.329] |
| 9 | -4.967 [-5.580, -4.363] | -1.179 [-1.753, -0.591] |
| 10 | -5.523 [-6.120, -4.906] | -1.617 [-2.164, -1.038] |
| 11 | -5.250 [-5.889, -4.596] | -1.232 [-1.799, -0.674] |
| 12 | -6.659 [-7.272, -6.039] | -2.882 [-3.429, -2.298] |
| 13 | -5.382 [-5.984, -4.762] | -1.996 [-2.561, -1.392] |
| 14 | -5.049 [-5.671, -4.412] | -1.069 [-1.642, -0.479] |
| 15 | -6.177 [-6.824, -5.523] | -3.030 [-3.632, -2.472] |
| 16 | -4.519 [-5.161, -3.883] | -0.495 [-1.071, +0.071] |
| 17 | -6.988 [-7.592, -6.400] | -3.083 [-3.622, -2.536] |
| 18 | -4.859 [-5.507, -4.211] | -0.874 [-1.461, -0.289] |
| 19 | -5.311 [-5.931, -4.677] | -1.720 [-2.287, -1.153] |
| 20 | -5.424 [-6.023, -4.822] | -1.668 [-2.206, -1.102] |

## Timing and faults

83,016,412 decisions; 6,784 deadline overruns (0.00817%) received check/fold fallback. Exceptions, illegal actions, internal errors and safety fallbacks were all zero. There were 55 separately recorded internal computation-budget fallbacks.

| Entrant | Decisions | Timeouts | Timeout rate (%) | Mean decision (Âµs) |
|---|---:|---:|---:|---:|
| Bot 1 | 3,117,297 | 47 | 0.00151 | 74.8 |
| Bot 2 | 3,256,958 | 95 | 0.00292 | 126.8 |
| Bot 3 | 3,359,053 | 44 | 0.00131 | 75.4 |
| Bot 4 | 3,577,494 | 44 | 0.00123 | 91.2 |
| Bot 5 | 3,411,047 | 87 | 0.00255 | 137.5 |
| Bot 6 | 3,480,088 | 119 | 0.00342 | 146.9 |
| Bot 7 | 3,561,443 | 121 | 0.00340 | 123.5 |
| Bot 8 | 3,739,275 | 167 | 0.00447 | 175.5 |
| Bot 9 | 3,483,490 | 38 | 0.00109 | 68.1 |
| Bot 10 | 3,373,446 | 36 | 0.00107 | 62.6 |
| Bot 11 | 3,550,504 | 80 | 0.00225 | 94.9 |
| Bot 12 | 3,451,670 | 48 | 0.00139 | 74.3 |
| Bot 13 | 3,536,372 | 124 | 0.00351 | 147.5 |
| Bot 14 | 3,816,041 | 89 | 0.00233 | 96.2 |
| Bot 15 | 3,568,411 | 59 | 0.00165 | 85.1 |
| Bot 16 | 3,609,146 | 76 | 0.00211 | 95.0 |
| Bot 17 | 3,248,822 | 47 | 0.00145 | 66.4 |
| Bot 18 | 3,519,375 | 65 | 0.00185 | 89.9 |
| Bot 19 | 3,461,595 | 4,601 | 0.13292 | 161.9 |
| Bot 20 | 3,619,835 | 68 | 0.00188 | 99.9 |
| Claude PPO | 6,698,509 | 359 | 0.00536 | 183.8 |
| Claude pure net | 6,576,541 | 370 | 0.00563 | 178.1 |

Bot 19 accounts for 4,601 overruns (0.133% of its decisions). Adapter conversion is timed. Startup/model loading is outside the decision clock. Shared hardware scheduling and time-budgeted actions prevent bit-identical behavior from seeds alone.

## Design, provenance and limits

- Pool-only:285 field instances per repetition, each bot in114 fields and each bot pair in42. Mixed:570 field instances per repetition, each bot in171 fields and each bot pair in45; both Claude baselines occur in every mixed field. Some compositions deliberately repeat to achieve exact balance.
- Four independently seeded repetitions; eight seat rotations; three payouts. Pool-only27,360 tables; mixed54,720. Each bot has3,648 pool-only and5,472 mixed tables per payout. Each baseline has18,240 mixed tables per payout.
- Bootstrap:5,000 replicates, resampling whole field-seed clusters with all seat rotations retained; ratio-of-sums/counts handles varying entrant presence. Marginal intervals are not adjusted for multiple comparisons and do not establish universal rankings.
- Experimental rules:8 seats,2,000 chips,10/20 initial blinds, doubling every25 hands,200-hand cap with ICM settlement. Payouts:top3(50/30/20),winner(100),top4(40/30/20/10). These are explicit local assumptions, not a verified official payout/blind schedule or interface.
- Both Claude variants are pinned to poker-prep commit a3d75f7c736da8c69e474ceed7e189682f9e6c7e. PPO uses the packaged ppo_v9_kl0.3/best submission; pure net uses policy_v9 CFR-distilled preflop policy with the same postflop models and default heuristic fallback. They are not pure full-game equilibrium solvers.
- Evaluation adapters now isolate absolute imports, expose opaque table IDs instead of deal seeds, decouple policy RNG from deck seeds, and reconstruct short-blind public history. Previous eight-bot experiments used different fields/harness and should not be directly compared as causal improvement estimates.
- All 12 new bots independently passed784 discovered regression tests, with10 documented optional skips and no failures. Original8 previously passed244 tests. Official integration remains unverified.

Reproducibility: [summary](pool20_summary.json), [schedule](schedule20.py), [runner](run_pool20.py), [analysis](analyze_pool20.py), [frozen bots](frozen_bots20.json), [harness hashes](pool20_harness_manifest.json). Raw shards and metadata will be retained in pool20_raw_results.tar.gz. Similarity findings are reported separately after their audit.
