# Evaluation results

Completed November 6, 2026 on MIT Engaging. Frozen bots were not tuned after seeing matchups.

## Eight-bot ranking

Ranked by mean prize-pool share with experimental 50/30/20 payouts. Each bot played 512 eight-only tournaments (64 seeds × eight seat rotations). Confidence intervals use 20,000 bootstrap resamples of whole seed blocks. Close ranks are not statistically established.

| Rank | Bot | Mean share | 95% interval | Difference vs Claude | 95% interval for difference |
|---:|---|---:|---|---:|---|
| 1 | bot_6 | 14.37% | 12.91–15.84% | -5.83 pp | -7.02 to -4.67 pp |
| 2 | bot_8 | 13.14% | 11.74–14.53% | -6.38 pp | -7.50 to -5.28 pp |
| 3 | bot_7 | 12.91% | 11.37–14.47% | -7.19 pp | -8.48 to -5.94 pp |
| 4 | bot_4 | 12.83% | 11.45–14.28% | -6.05 pp | -7.15 to -4.95 pp |
| 5 | bot_2 | 12.64% | 11.31–13.96% | -7.69 pp | -8.86 to -6.53 pp |
| 6 | bot_5 | 11.95% | 10.62–13.34% | -7.40 pp | -8.60 to -6.21 pp |
| 7 | bot_3 | 11.17% | 9.32–13.09% | -8.05 pp | -9.32 to -6.78 pp |
| 8 | bot_1 | 10.98% | 9.57–12.38% | -8.75 pp | -9.94 to -7.60 pp |

The Claude differences use 3,584 tables per bot per payout where that bot and Claude both played, with identical table outcomes paired before seed-block aggregation. They are not heads-up matches or a comparison to seven Claude clones. Claude beat every bot by this measure under all three payout schedules; every marginal 95% difference interval was below zero.

## Payout sensitivity

- **50/30/20:** bot_6 > bot_8 > bot_7 > bot_4 > bot_2 > bot_5 > bot_3 > bot_1.
- **Winner takes all:** bot_2 > bot_4 > bot_6 > bot_3 > bot_5 > bot_8 > bot_7 > bot_1.
- **40/30/20/10:** bot_4 > bot_6 > bot_5 > bot_8 > bot_2 > bot_7 > bot_3 > bot_1.

These are point-estimate orders, not proven strict rankings. With unknown official payouts there is no defensible single universal winner. Bot 6 led the primary payout experiment; bot 2 led winner-take-all; bot 4 led top-four payouts.

## Design and audit

- 13,824 tournaments: 64 seeds × nine leave-one-out fields × eight rotations × three payout schedules. The field consists of our eight bots and the Claude baseline; omitting Claude gives the eight-only ranking.
- All matches use 2,000 starting chips, 10/20 initial blinds, blind doubling every 25 hands, and a 200-hand cap with ICM settlement. Payouts and blind increments are experimental assumptions.
- All 24 production shards completed successfully. Source/model SHA-256 checks passed on every shard; bot 1’s native Linux accelerator loaded.
- Complete, unique, seat-balanced schedule; chip and prize conservation verified. Zero recorded exceptions, illegal actions, or reported internal errors.
- 244 original export tests passed, plus 15 common-engine/adapter tests. The shared engine derives from the pinned baseline repository; a short-all-in blind call-cost correction is tracked separately from the frozen baseline.
- Adapter conversion is included in each bot’s 1 ms decision window. Imports and initialization happen before timing. Late decisions become check/fold.
- Confidence intervals reflect seed variation, not all possible opponents, hardware, engine choices, or official contest conditions. They are marginal intervals, without multiple-comparison correction.
- Engaging used heterogeneous CPU nodes. Wall-clock scheduling and time-budgeted sampling can change actions even with identical deck seeds. Timing numbers are observations, not a universal latency guarantee.

## Timing and fallback

| Bot | Decisions | Over 1 ms | Fallback rate | Mean decision time |
|---|---:|---:|---:|---:|
| Claude | 1,506,283 | 2,011 | 0.1335% | 193.8 µs |
| bot_1 | 1,381,291 | 20 | 0.0014% | 71.7 µs |
| bot_2 | 1,437,006 | 112 | 0.0078% | 122.8 µs |
| bot_3 | 1,493,459 | 15 | 0.0010% | 68.9 µs |
| bot_4 | 1,631,578 | 56 | 0.0034% | 84.0 µs |
| bot_5 | 1,549,275 | 44 | 0.0028% | 130.1 µs |
| bot_6 | 1,592,506 | 45 | 0.0028% | 142.0 µs |
| bot_7 | 1,575,706 | 69 | 0.0044% | 114.7 µs |
| bot_8 | 1,673,707 | 105 | 0.0063% | 166.6 µs |

## Reproduction

- Baseline: `A1exL1ang/poker-prep`, commit `a3d75f7c736da8c69e474ceed7e189682f9e6c7e`, `vibe_coding_exp/claude_exp_1/submission/submission_config.json`.
- Frozen configuration: PPO `ppo_v9_kl0.3/best`, postflop `postflop_v6s`, three-player postflop `postflop3_v1`; no heuristic overrides.
- `frozen_bots.json` and `baseline_manifest.json`: original source/model hashes.
- `run_matches.py`, `adapters.py`, `common/`: shared experimental protocol and engine.
- `production.slurm`: bounded preemptible job configuration; no existing user allocations were used.
- `production_results.jsonl.gz`: all raw production matches. `summary.json`: audited aggregate estimates and intervals.
- `analyze.py`: validates completeness and computes the summary. To rerun analysis, split the decompressed matches into files named `production_*.jsonl` under `evaluation/results`, then run `python evaluation/analyze.py`.
- Production array `28067659` completed; the earlier array `28067291` failed before matches because the manifest verifier needed Windows-path normalization. This was fixed without editing any bot.

No official starter kit was available. This evaluation does not establish official-interface compatibility or predict contest results.
