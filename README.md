# Poker pool

Eight independent no-limit Texas Hold'em bots, built concurrently on November 6,
2026. Each builder had the contest rules and the objective of maximizing its
chance of winning. Builders used separate repositories and contexts, with
instructions prohibiting access to other bots. This was instructional isolation,
not an OS-level read barrier. Fast mode was disabled for all eight builds.

| Directory | Builder | Local tests passed |
|---|---|---:|
| bot_1 | Sol high | 39 |
| bot_2 | Sol xhigh | 24 |
| bot_3 | Sol max | 36 |
| bot_4 | Astra high | 36 |
| bot_5 | Astra high | 23 |
| bot_6 | Astra xhigh | 39 |
| bot_7 | Astra xhigh | 23 |
| bot_8 | Astra max | 24 |

Each directory contains its own implementation, tests, and documentation. All
eight builds finished within one hour; exported tests were independently rerun.
Their separate local benchmarks are **not a ranking**.

The official starter kit was unavailable. These bots implement documented local
interfaces; official contest compatibility and a strict 1 ms timing guarantee
have not been established.

## Evaluation results

**13,824 Engaging tournaments completed.** With experimental 50/30/20 payouts,
the eight-only point-estimate order was **6, 8, 7, 4, 2, 5, 3, 1**. Several
confidence intervals overlap. The leader changed to bot 2 with winner-take-all
payouts and bot 4 with 40/30/20/10 payouts. The pinned Claude baseline outscored
every bot in matched mixed-table comparisons under all three payout schedules.

See [full results and uncertainty](evaluation/RESULTS.md),
[machine-readable summary](evaluation/summary.json), and
[raw matches](evaluation/production_results.jsonl.gz).

`evaluation/` contains the frozen source hashes, baseline provenance, common
engine, per-bot adapters, and seeded tournament runner. The baseline is the packaged submission from
`A1exL1ang/poker-prep/vibe_coding_exp/claude_exp_1`, pinned to
`a3d75f7c736da8c69e474ceed7e189682f9e6c7e`. Its code is obtained separately, not
vendored as a second repository here. The common engine and its initial tests
are derived from that revision, with evaluation changes tracked here.

Experimental games use eight players, 2,000 chips each, 10/20 initial blinds,
blind doubling every 25 hands, and a 200-hand cap with ICM settlement. The blind
increments and payouts are experimental assumptions. Evaluated payout sensitivity:
50/30/20, winner-take-all, and 40/30/20/10. Every seed rotates all eight seats;
nine leave-one-out fields cover the eight local bots plus the Claude baseline.
Decision timing includes adapter conversion and uses check/fold fallback above
1 ms. Timing-dependent policies and operating-system scheduling mean seeded
decks do not guarantee bit-identical actions.

Run local evaluation checks after obtaining the pinned baseline and installing
NumPy and pytest:

```powershell
$env:PYTHONPATH='evaluation'
python -m pytest evaluation/tests -q
```

Engaging runs must use new bounded preemptible allocations. Existing user jobs
are never used or modified. Private worker logs and SSH session details are
excluded from this repository.
