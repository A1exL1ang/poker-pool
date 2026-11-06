# Pure-net Claude comparison

13,824 additional eight-player tournaments on Engaging, November 6, 2026. The eight original bots were frozen. The baseline used CFR-distilled `policy_v9` with `postflop_v6s` and `postflop3_v1`, no PPO fine-tuning, no supplied heuristic overrides. The original default heuristic fallback remains outside supported net situations.

## Matched comparison against each bot

50/30/20 payouts. Positive numbers below mean the pure-net Claude baseline earned more prize-pool share. Each comparison covers 3,584 tables containing both players; uncertainty resamples 64 whole seed blocks, including all seat rotations and relevant opponent fields. These are mixed eight-player tables, not heads-up games.

| Opponent | Net advantage (percentage points) | 95% interval |
|---|---:|---|
| bot_1 | +3.76 | +2.71 to +4.88 |
| bot_2 | +2.41 | +1.29 to +3.52 |
| bot_3 | +3.16 | +2.13 to +4.21 |
| bot_4 | +0.51 | -0.41 to +1.42 |
| bot_5 | +1.95 | +0.83 to +3.13 |
| bot_6 | +0.71 | -0.32 to +1.76 |
| bot_7 | +1.57 | +0.34 to +2.86 |
| bot_8 | +1.66 | +0.66 to +2.66 |

## Pure net versus previous PPO experiment

Average baseline share across balanced mixed fields; 4,096 tables per baseline per payout. The contrast uses matching seed blocks across separate experiments, not tables containing both baseline variants.

| Payout | Pure net share | PPO share | Net minus PPO (pp), 95% interval |
|---|---:|---:|---|
| 50/30/20 | 14.22% | 18.77% | -4.55 [-5.38, -3.72] |
| Winner takes all | 15.72% | 19.56% | -3.83 [-5.18, -2.51] |
| 40/30/20/10 | 14.31% | 17.98% | -3.66 [-4.32, -3.00] |

## Audit and limits

- All 24 shards completed; frozen source/model hashes verified on every shard. All matches have unique complete schedule IDs, correct chip/prize totals, and zero recorded exceptions, illegal actions or internal errors.
- Identical experimental engine, initial stacks, blinds, payout sensitivity and 1 ms check/fold fallback as the prior experiment. See `RESULTS.md` for full design.
- Source pinned to `a3d75f7c736da8c69e474ceed7e189682f9e6c7e`. `net_baseline_manifest.json` contains selected weights/configuration and hashes; `build_net_baseline.py` reconstructs it from that repository.
- The pure-net models use solver-generated targets. This is not a claim of exact full-game equilibrium or absence of heuristic runtime choices.
- Hardware nodes and timing-dependent action sampling vary; seed pairing controls deck schedules, not all environmental randomness. Marginal 95% intervals are not adjusted for multiple comparisons.
- No official starter kit or payout schedule was available. Results apply to these experimental fields and rules.

## Timing

| Bot | Decisions | Over 1 ms | Fallback rate |
|---|---:|---:|---:|
| Claude pure net | 1,485,688 | 10,817 | 0.7281% |
| bot_1 | 1,403,654 | 25 | 0.0018% |
| bot_2 | 1,459,581 | 246 | 0.0169% |
| bot_3 | 1,515,236 | 49 | 0.0032% |
| bot_4 | 1,659,306 | 117 | 0.0071% |
| bot_5 | 1,574,386 | 78 | 0.0050% |
| bot_6 | 1,617,718 | 64 | 0.0040% |
| bot_7 | 1,591,874 | 139 | 0.0087% |
| bot_8 | 1,690,670 | 237 | 0.0140% |
