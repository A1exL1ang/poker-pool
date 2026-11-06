# bot_17 — independent offline NLHE candidate

Python bot, independently authored simulator, equity engine, generated assets,
tests, and reproducible experiments. This is local development, **not a contest
submission**. No official starter kit was available. `CONTRACT.md` defines our
local observation/action protocol; an official adapter and fresh timing tests
are required when the real interface arrives.

Final local evidence: 73 tests, 8,500 held-out tables, 1,201,464 decisions,
zero errors/illegal actions and four enforced timeout fallbacks. The remote
native 50,000-call stress test had p99 189 µs and zero overruns; local Windows
and WSL runs did have overruns. See [the full report](measurements/REPORT.md)
for settings, confidence intervals and limitations.

## Run

Python 3.10+; runtime has no Python package dependencies and makes no network
calls. Import and model loading happen before decisions.

```python
from agent import Agent
bot = Agent(seed=17)
action = bot.act(observation)  # CONTRACT.md: raises are TOTAL street amounts
bot.reset()                  # reset at every new table
```

`python run_bot.py` accepts one JSON observation per input line and writes one
JSON action. `{"command":"reset","seed":17}` resets table state. Process start
and JSON transport are not part of the reported in-process decision timing.

The native accelerator is optional: `native/poker_equity.dll` for Windows x64
and `native/poker_equity.so` for Linux x64. Its source is included. Rebuild with
`python tools/build_native.py` using a C++17 compiler. If loading fails, the bot
falls back automatically to a bounded Python equity model; inspect
`equity.BACKEND` and `equity.NATIVE_ERROR`. Set `POKER_PURE_PYTHON=1` before import
to test without native code. Native binaries are architecture dependent.

## Strategy

- Position-dependent preflop opens, stack-dependent shoves, value reraises,
  occasional suited-ace reraises, priced calls, and restrained set mining.
- Preflop equity lookup generated from this bot's own Monte Carlo engine:
  169 hand classes, seven opponent counts, seven range widths, 20,000 trials
  per cell. Opponent ranges use a documented heuristic ordering. These are
  estimated equities, not equilibrium ranges or externally supplied data.
- Postflop equity accounts for card removal, ties, multiple opponents,
  preflop range narrowing, and board-dependent betting likelihoods. The
  policy uses effective pot odds, position, draws, value bets and mixed bluffs.
  Heads-up river equity is enumerated exactly within the modeled range.
  Multiway side pots share a sampled deal and have separate eligibility masks.
- Small anonymous-seat statistics with substantial priors adapt only within
  a table. No team identification or cross-table opponent memory.
- Exact subset-DP ICM is used for large heads-up calls **only when actual
  payouts are supplied**. Otherwise the policy uses chip odds and explicit
  strategic margins. No official payout vector is invented.
- Legal-action normalization handles raise-to totals, all-in underraises,
  reopening permissions, capped calls, and safe check/fold input fallback.

This is a compact heuristic policy with range equity, not a full-game Nash/CFR
solution. Range inference, multiway action EV, and future equity realization
remain approximations. Betting likelihoods and hand rankings were authored
locally. The small simulation opponent pool cannot represent unknown entrants.

The Python fallback uses 160 depth-six boosted trees trained on 100,000 locally
generated states, with 6,000 separate validation states. Four or more distinct
side-pot masks use the first 80 trees for timing headroom. It predicts each
pot from average range parameters and omits some blocker correlations. A
4,000-state held-out estimator audit measured RMSE 0.0492, versus 0.0342 for
the native runtime estimator, against our high-sample model. Rare fallback
errors were large (maximum about 0.57 in the side-pot audit); native is preferred.
These tests measure the declared range model, not actual opponent calibration.
Multiway ranges are sampled sequentially with card removal; this approximates
the fully conditioned joint range distribution. Bounded rejection can rarely
fall back to the last valid proposal or a uniform legal hand. These events are
counted by `equity.sampler_stats()` and reported with the measurements.

## Rules and local assumptions

The complete supplied `RULES.txt` was read. The local engine uses eight initial
seats, 2,000 chips each, 10/20 starting blinds, levels lasting 25 hands, and a
200-hand limit. Blinds after the initial level are configurable **experimental
assumptions**. No ante or rake is assumed. Payout vectors named `top3`, `winner`
and `linear` are illustrative sensitivity tests, not official contest points.
Survivor point allocation uses ICM at the hand cap. Same-hand eliminations
with equal starting stacks split the occupied place points locally; that tie
rule also needs official confirmation.

The engine follows standard NLHE best-five evaluation, burn/deal order,
heads-up blind/action order, full-raise and cumulative-short-all-in reopening,
uncalled-chip refunds, per-pot eligibility, odd chips left of the button,
and dead-button progression. Observations reveal no opponent hole cards or
future board cards. Bot exceptions/invalid actions use check when possible,
otherwise fold. `--enforce-time` applies the same fallback after a measured
wall-clock decision exceeds 1 ms.

## Reproduce

```text
python -m unittest discover -s tests -v
python tools/build_native.py
python tools/generate_preflop.py --samples 20000 --workers 8 --ranges 0,.5,.75,.85,.9,.95,1
python tools/train_fallback.py --train-rows 100000 --validation-rows 6000 --trees 160 --depth 6 --train-samples 256 --validation-samples 2048
python tools/heldout_fallback.py --spots 4000 --samples 4096
python tools/heldout_sidepots.py --spots 600
python tools/equity_benchmark.py --calls 6000 --samples 64 --output measurements/equity_latency.json
python tools/experiment.py --phase tune --tables 150 --workers 8
python tools/experiment.py --phase confirm --profiles base,cautious,call_tight --tables 500 --workers 8
python tools/experiment.py --phase heldout --profiles base --tables 1000 --workers 6 --enforce-time
python tools/decision_benchmark.py --calls 50000 --output measurements/decision_latency.json
python tournament.py --tables 100 --seed 1500000 --label independent_recheck --lineup mixed --payouts top3 --enforce-time --output measurements/recheck.json
python tools/package.py
python tools/verify_package.py
```

Tuning/confirmation/evaluation seeds are disjoint and explicit. Profile and
environment results, source hashes, per-table outcomes, confidence intervals,
errors, invalid actions, internal fallbacks, and observed/enforced timeouts are
saved under `measurements/`. A timing measurement depends on machine load;
low median/p99 latency does not guarantee a hard deadline. Import time is
reported separately from warm decisions. Detailed measurements are recorded
in `measurements/REPORT.md`. The selected base profile and selection rationale
were saved in `measurements/selection.json` before tournament evaluation.

`dist/bot_17.zip` is the offline runtime artifact; `MANIFEST.json` records file
hashes. Source, benchmark scripts and tests remain in the workspace. `.control`
contains operational metadata and our own compute job ledger; it is excluded
from the runtime archive. Nothing is pushed, published, or submitted.

## Public research consulted

- [Supplied contest rules source](https://pokerbot.sequoiacap.com/rules.html):
  the local `RULES.txt` is the actual rules input for this task.
- [Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/) and
  [TDA administrator clarification of short opening all-ins](https://www.pokertda.com/forum/index.php?topic=1456.0):
  mechanics references, including cumulative reopening and minimum raises.
- [PokerStars rules explanation](https://www.pokerstars.com/help/articles/poker-rules-master/):
  heads-up order and no-limit raising reference.
- [Brown and Sandholm, *Superhuman AI for multiplayer poker*, 2019](https://noambrown.github.io/papers/19-Science-Superhuman.pdf):
  research context for mixed strategies and multiplayer search. We do not
  implement Pluribus or claim its performance/theoretical properties.

No other bot workspace, private poker repository, or team's artifact was used.
