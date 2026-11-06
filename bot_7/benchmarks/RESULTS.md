# Bot 7 validation

Generated 2026-11-06T06:39:37.264509+00:00.

All final tournament and latency reports match runtime SHA-256 `0a03c81125b4cf933fa1612e82a77ad0fbe7da2724f27dc3defd6e9c8848f054`. No official environment or competitors were available.

23 unit tests passed. The independent evaluator audit checked all 2,598,960 five-card category counts; 7,500 random five-, six-, and seven-card hands matched a separate slow best-five evaluator, including tie breakers. Exact ICM was checked against permutations, independent eight-player subset DP, symmetry, and utility conservation. Engine regressions cover side pots, short all-ins, reopening, heads-up blinds, and a short-big-blind phantom-call edge case.

On Windows-10-10.0.26200-SP0, Python 3.11.9, 30,000 direct calls over 1,616 recorded/stress states measured:

| Boundary | Median | p95 | p99 | Maximum |
|---|---:|---:|---:|---:|
| Agent.act | 0.170 ms | 0.387 ms | 0.540 ms | 11.439 ms |
| Dictionary conversion + act | 0.220 ms | 0.442 ms | 0.577 ms | 0.964 ms |

**The strict 1 ms wall-clock requirement is not guaranteed:** 48/30,000 direct calls exceeded it (0.160%); the separate adversarial/stress repetitions had 36/12,000 overruns. The 450 microsecond soft budget reduces sampling work but cannot prevent scheduler/GC delays or bound the official adapter. All raw outliers are retained. Aggregate thread CPU usage, including harness overhead, averaged 0.172 ms/call. Per-call CPU percentiles are not reported because Windows accounting was too coarse.

Timing excludes cold import. A separate local cold-import observation was approximately 0.261 s; assets must load before timed decisions if the official environment permits that lifecycle.

All tournaments below are local synthetic tests: eight initial seats, 2,000 chips, at most 200 hands, and big blinds 20/30/50/80/120/200/320/500, changing every 25 hands. No ante. Finishing tie convention uses chips at hand start. These are test choices, not the missing official progression or points.

Win-equity is final chip fraction at the cap, equivalently winner-take-all ICM utility; it is not the observed frequency of outright victories. The linear-point column uses hypothetical payouts 8/7/6/5/4/3/2/1. A symmetric reference is 12.5% win-equity and 4.5 linear points. SE is the standard error across seeded tables, not a bound on performance against unknown competition bots.

| Fixture | Tables | Mean win-equity ± SE | Hypothetical linear points ± SE | Hero errors |
|---|---:|---:|---:|---:|
| [Mixed simple opponents](mixed_final.json) | 500 | 42.202% ± 2.037% | 6.177 ± 0.087 | 0 |
| [Equity-based simple opponent](equity_final.json) | 300 | 42.107% ± 2.807% | 6.578 ± 0.106 | 0 |
| [Identical-policy clones](self_final.json) | 300 | 12.695% ± 1.829% | 4.516 ± 0.133 | 0 |
| [Mixed, timeout fallback enforced](timeouts_final.json) | 200 | 43.514% ± 3.227% | 6.225 ± 0.139 | 0 |
| [Mixed, hypothetical 50/30/20 payout input](payouts_final.json) | 100 | 47.202% ± 4.629% | 6.577 ± 0.168 | 0 |

The final runs total 1,400 tables, 238,439 hands, and 233,515 timed hero actions. Chip conservation and action legality are asserted throughout. The first three strategy experiments disable the soft clock budget for deterministic seeded decisions. The timeout and payout-input runs use production defaults, so timings can change their trajectories. Timeout fallback applies to every seat. These opponent fixtures are weak and share utility/evaluation helpers with the test harness; they are not independent contest-strength rivals.

River sampling was compared with exact enumeration of the declared heuristic ranges on 120 states and 80 seeds per state: signed bias +0.00211, single-decision RMSE 0.04681, and mean absolute error of the 80-estimate averages 0.00549. This measures sampling error, not correctness of opponent range assumptions. See [equity_audit.json](equity_audit.json).

The expanded tournament validation found a local simulator error when a short big blind left only one player able to act: it could demand a nonexistent full-blind completion above every opposing all-in. This was fixed, directly regression-tested, and excluded from final results by rerunning the final benchmarks. Older exploratory files are retained but are not the final evidence.

The local package is generated and isolated-smoke-tested by `python -m tools.package`; its size and hashes are in `../build/manifest.json`. No external compute jobs, submissions, network gameplay, commits, or publication were performed. The unknown official interface, payout/blind schedule, startup accounting, and target-hardware latency remain integration limitations. See [README](../README.md) and [interface contract](../INTERFACE.md).
