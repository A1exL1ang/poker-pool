# Poker pool

Twenty independent no-limit Texas Holdâ€™em bots, built November 6, 2026. Each builder received the contest rules, the objective of maximizing its chance of winning, and standard no-limit Holdâ€™em. Builders used separate repositories and contexts with strict no-cross-read instructions. This was instructional isolation, not an OS access barrier. Fast mode was off.

The first eight builds had a one-hour limit. Twelve additional builds had no fixed deadline and could use up to two private helpers each. All 20 policies were frozen before common evaluation. No builder received another botâ€™s strategy or the Claude baseline.

## Latest results: all 20 bots

**82,080 balanced eight-seat tournaments completed on Engaging.** Bot 18 leads the top3/top4 point estimates; bot 6 leads winner-take-all. Bot 18 and bot 6 are effectively tied under50/30/20 payouts (13.742% versus13.725% prize share), with overlapping uncertainty intervals.

The packaged **Claude PPO baseline outscored every bot** in matched same-table comparisons under all three payout shapes. PPO also beat the non-PPO CFR-distilled net directly by4.04 percentage points under50/30/20 payouts (95% CI3.62â€“4.47). These are multiplayer field-specific results, not universal or heads-up strength guarantees.

- [Complete rankings, baseline comparisons, timing and limitations](evaluation/POOL20_RESULTS.md)
- [Machine-readable audited summary](evaluation/pool20_summary.json)
- [Raw match shards and metadata](evaluation/pool20_raw_results.tar.gz)
- [Frozen bot hashes](evaluation/frozen_bots20.json) and [harness hashes](evaluation/pool20_harness_manifest.json)

All 83,016,412 decisions were audited: zero exceptions, illegal actions or safety fallbacks;6,784 strict1 ms timeout fallbacks and55 separate internal budget fallbacks. All 12 new exports passed independent regression runs:784 tests discovered,10 optional skips,zero failures. The original eight previously passed244 tests. Neither official-interface compatibility nor a universal1 ms guarantee is claimed.

## Builders

| Bot | Model and effort |
|---:|---|
| [1](bot_1) | Sol high |
| [2](bot_2) | Sol xhigh |
| [3](bot_3) | Sol max |
| [4](bot_4) | Astra high |
| [5](bot_5) | Astra high |
| [6](bot_6) | Astra xhigh |
| [7](bot_7) | Astra xhigh |
| [8](bot_8) | Astra max |
| [9](bot_9) | Sol high |
| [10](bot_10) | Sol high |
| [11](bot_11) | Sol xhigh |
| [12](bot_12) | Sol xhigh |
| [13](bot_13) | Sol max |
| [14](bot_14) | Sol max |
| [15](bot_15) | Astra high |
| [16](bot_16) | Astra high |
| [17](bot_17) | Astra xhigh |
| [18](bot_18) | Astra xhigh |
| [19](bot_19) | Astra max |
| [20](bot_20) | Astra max |

Sol denotes gpt-6.1-sol; Astra denotes gpt-6-astra. Each bot directory preserves its own implementation, documentation, tests and experimental evidence. Separate builder opponent suites are not comparable rankings.

## Evaluation and reproduction

The20-only design uses285 field instances per repetition; mixed comparisons use570 fields of6 bots plus both Claude variants. Four independent repetitions,8 seat rotations and3 payouts produce82,080tables. Each bot and pair has exactly balanced exposure within each cohort; deliberate composition multiplicities are documented in [schedule20.py](evaluation/schedule20.py). Confidence intervals resample field-seed clusters with rotations retained.

Local experimental assumptions:8 seats,2,000 chips,10/20initial blinds, doubling every25 hands,200-hand cap with ICM settlement; payouts50/30/20,100,and40/30/20/10. The official starter kit and complete payout/blind schedule were unavailable. Conversion is inside the1 ms decision clock. OS scheduling and internal time budgets can change actions despite fixed seeds.

Both Claude variants are pinned to poker-prep commit `a3d75f7c736da8c69e474ceed7e189682f9e6c7e`. The PPO policy is the packaged `vibe_coding_exp/claude_exp_1/submission`; the non-PPO version uses CFR-distilled `policy_v9` with the same postflop nets and default heuristic fallback. External baseline code is obtained separately, not vendored here. The common engine derives from that revision with tracked evaluation fixes.

After obtaining the pinned baselines and installing NumPy, run on an appropriate compute allocation:

```sh
export PYTHONPATH=evaluation
python evaluation/verify_pool20.py
python evaluation/verify_bundle.py --net
python -m unittest discover -s evaluation/tests -v
python -m unittest discover -s evaluation -p 'test_*.py' -v
python evaluation/run_pool20.py --pilot --output evaluation/pilot.jsonl
python evaluation/run_pool20.py --shard 0 --shards 24 --output evaluation/pool20_results/production_0.jsonl
python evaluation/analyze_pool20.py --results evaluation/pool20_results --output evaluation/pool20_summary.json
```

Run all 24 shards before analysis. [pool20.slurm](evaluation/pool20.slurm) records bounded resource settings. Use only new task-owned preemptible jobs; existing user allocations remain untouched. Private worker logs, SSH details and development virtual environments are excluded.

## Earlier eight-bot experiments

Preserved: [PPO comparison](evaluation/RESULTS.md), [pure-net comparison](evaluation/NET_RESULTS.md), [PPO raw matches](evaluation/production_results.jsonl.gz), [pure-net raw matches](evaluation/net_results.jsonl.gz). These used different fields and an earlier adapter harness; do not interpret cross-run differences as isolated policy improvements.

See [bot similarity findings](evaluation/SIMILARITY.md) for source structure, action agreement and bet-sizing comparisons.
