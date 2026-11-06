# bot_5 validation results

These are local development results for one independent bot. They do not establish official interface compatibility or predict contest standings.

## Correctness

- **23 tests passed on Windows Python 3.11 and Linux Python 3.12.** Logs: `tests.txt`, `linux_tests.txt`.
- Independent evaluator: 5,000 sampled five-, six-, and seven-card hands agree with exhaustive five-card-subset comparison using a separately written reference evaluator.
- 3,000 shared-board evaluator cases; 3,000 random canonical flop lookups with suit relabeling; all 1,286,792 asset keys strictly increasing; no missing sampled states.
- 1,500 randomized legal-state checks. Side pots, uncalled excess, odd chips, short all-in reopening, short big blinds, heads-up order, table resets, shared public nuts, and chip conservation are covered.
- Exact ICM checked against independent finish-order enumeration.
- All **2,598,960 five-card hands** reproduce the combinatorial category census; see `exhaustive_evaluator.json`.

## Complete decision latency

Asset loading and transport are outside the timed `Agent.act` call. Validation, history updates, payout-aware risk, and strategy are inside it. Wall-clock measurements are reported without dropping outliers.

| Environment | Calls | Median | p99 | Maximum | Calls over 1 ms |
|---|---:|---:|---:|---:|---:|
| Windows 11, Python 3.11.9 | 28,512 | 0.116 ms | 0.512 ms | 18.729 ms | 68 |
| Engaging Linux, Python 3.12.7 | 27,807 | 0.072 ms | 0.227 ms | 0.575 ms | 0 |
| Linux timeout-enforced tournaments | 37,707 | 0.071 ms | 0.223 ms | 0.495 ms | 0 |

The Windows run uses 9,504 distinct game observations, each replayed three times. Linux uses 9,269 distinct observations, also replayed three times. Repeated states are not independent timing trials. Windows thread CPU timers were too coarse to interpret per-call CPU use; the raw wall-clock tails remain a limitation. Asset initialization alone took about 4.9 ms on Windows and 10.0 ms on Linux; module import/startup is not included. An official sandbox could account for startup or I/O differently.

Raw reports: `latency.json`, `linux_latency.json`, and `linux_timeout_simulation.json`. The earlier `latency_profile.json` and `benchmark_corpus.jsonl` retain the first held-out profile and its input corpus. JSON parsing/encoding is measured separately inside the latency reports.

Reproduce the final Windows run:

```text
python tools/benchmark.py --tables 64 --repeats 3 --seed 191000 --output evidence/latency.json
```

The Linux script is `tools/remote_validate.sh`. It ran as **bot_5_validate**, Slurm job **28063387**, in this bot's private remote directory, with one newly requested CPU, 1 GB memory, and an eight-minute preemptible allocation. It completed successfully in **20 seconds**. No existing user allocation was used, and no bot_5 job remains running. The completed job log is `remote_validation_28063387.log`.

## Equity accuracy

The flop asset has 1,024 trials per opponent range at every canonical state. Fresh independent-seed reference simulation checked 400 randomly selected states against all four ranges, with 20,000 reference trials per range:

| Metric | Absolute equity error |
|---|---:|
| Mean | 1.046 percentage points |
| RMS | 1.329 points |
| 95th percentile | 2.688 points |
| Maximum | 5.189 points |

The reference is Monte Carlo, not exact equity. See `flop_validation.json`. This validates stored heads-up range equities, not the later multiway, betting, or payout heuristics.

## Synthetic tournament suite

Each table starts eight players with 2,000 chips. Test blinds double from 20 every 25 hands. Test points are **0.5/0.3/0.2/0/0/0/0/0**; the equal-share benchmark is 0.125. These later blinds and payouts are deliberately synthetic, not official. Hero seat rotates. No timing penalties are applied to this suite; timeout behavior is tested separately on Linux.

| Seven opponents | Tables | Mean points | Approximate 95% interval |
|---|---:|---:|---:|
| Mixed simple baselines | 1,024 | 0.3425 | 0.3320–0.3531 |
| Tight baseline | 256 | 0.2754 | 0.2500–0.3007 |
| Loose-passive baseline | 256 | 0.4095 | 0.3872–0.4318 |
| Aggressive baseline | 256 | 0.3387 | 0.3224–0.3550 |
| Random baseline | 256 | 0.3633 | 0.3398–0.3867 |
| Copies of this bot | 128 | 0.1422 | 0.1097–0.1747 |

All **2,176 tables** completed without policy errors or illegal-action failures. The baselines are intentionally simple and weak; these results demonstrate functional play and a substantial advantage over those baselines, not over unknown competition bots. Self-play is consistent with equal expected shares. Intervals use the normal approximation and sample standard error.

The separate Linux timeout-enforced run used 256 mixed tables, seed 182000, and scored **0.3270 ± 0.0224** (approximate 95% interval half-width), with no timeout penalties actually triggered. Reproduction: `python tools/tournament_suite.py`; raw table records live in `final_*.json` and `tournament_suite.json`.

## Delivery

`bot_5_runtime.zip` contains only runtime Python, four generated assets, their manifest, the README, and an example input. It is approximately **6.63 MB compressed / 11.24 MB uncompressed**, well below the 100 MB file limit. `package.json` records exact final sizes, archive SHA-256 and every runtime source hash. All build and test sources remain in the workspace. No competition submission or publication was performed.
