# bot_15 — independent offline Texas Hold’em bot

This workspace contains one Python bot, its locally defined interface, an independent simulator, tests, training sources, and experiment records. It is a local experimental deliverable, **not a contest submission**. No official starter kit was available; compatibility with an unseen official `Agent` interface is not claimed. The collector must map the real interface to [INTERFACE.md](INTERFACE.md), supply its legal actions and actual payouts, and remeasure timing in its sandbox.

The final gameplay archive is [dist/bot_15.zip](dist/bot_15.zip). See [MEASUREMENTS.md](MEASUREMENTS.md) for the complete held-out results, actual overrun counts, and limitations. `artifacts/package_verification.json` records archive size, hash, and offline verification.

## Run

Python 3.10+ and the standard library are sufficient for gameplay. No network, API, model service, native extension, NumPy, or scikit-learn is used in gameplay.

```python
from agent import Agent
bot = Agent(seed=15)       # load/validate immutable assets before decision timing
action = bot.act(observation)
bot.reset()               # REQUIRED at every new table
```

`python agent.py` also accepts one local observation JSON object per input line. `{"reset":true}` resets table state. This CLI is a smoke/integration utility; JSON parsing, process startup, and I/O are not included in the in-process `act` benchmark.

## Rules and integration assumptions

`RULES.txt` was read in full. The bot implements the user's disclosed standard no-limit Texas Hold’em variant. The contest specifies eight players, 2,000 initial chips, opening blinds 10/20, increases every 25 hands, a 200-hand cap, finishing points with ICM at the cap, table-state resets, offline execution, 1 ms decisions, and a 100 MB file limit.

The actual interface, complete blind progression, payout vector, initialization budget, CPU model, and timing instrumentation are unknown. The local simulator defaults to **doubling** blinds every 25 hands; that progression is an experimental assumption. All supplied benchmark point vectors are explicitly illustrative. No official points are invented. Without payouts the agent uses chip expected value; with payouts it can apply ICM risk adjustments. An optional `position` supplied by the adapter is needed for precise preflop ordering with dead blinds; the engine supplies it.

Card IDs are rank-major integers `0..51`: `rank=card//4`, with 2 through Ace mapped to `0..12`; `suit=card%4`. A raise amount is the **total commitment on this street**, while a call amount is newly paid chips capped by stack. Pot includes all chips already committed, including live bets. All-in players remain in `in_hand`. `can_raise` and `min_raise_to` must reflect action reopening. These details are rule-critical; do not guess them from a different API.

## Strategy

The independently designed policy combines combo-weighted preflop ranges, position and effective-stack sizing, a short-stack shove approximation, pot odds, pot-layer eligibility, and a trained postflop heads-up equity surrogate. It mixes limited bluffs and value bets. Bayesian seat-local statistics weakly adapt ranges and aggression estimates from observed action histories. No team identities or cross-table statistics are retained.

The model estimates equity against four strengths of preflop ranges. A soft range-strength adjustment approximates postflop betting evidence. Multiway equity and future equity realization are approximations; the policy is not a GTO solver or a game-theoretic guarantee. Exact forced-chop rules override learned estimates. The exact subset ICM calculator is independently checked, but the policy's multiway fold/win/lose ICM scenarios are simplified and capped.

## Validation and reproduction

```text
python -m unittest discover -s tests -v
python -m benchmarks.latency --tables 100 --repeat 4
python -m benchmarks.fuzz --help
python -m benchmarks.tournament --tables 1000 --split heldout --paired balanced --payouts 50,30,20,0,0,0,0,0
python -m training.tune_policy --tables 128 --workers 8
```

The native evaluator audit exhaustively checked all 2,598,960 five-card hands and their canonical category counts, plus 10,000 random seven-card hands against enumeration of independent five-card ranks. Native/Python features matched on 19,664 cases including every four-of-a-kind turn board with every legal hole pair. The simulator additionally checks blind ordering, incomplete and cumulative all-ins, reopening, capped calls, side pots, uncalled excess, odd chips, simultaneous eliminations, and chip/point conservation. See [benchmarks/README.md](benchmarks/README.md) and [training/equity_README.md](training/equity_README.md).

Equity training uses 240,000 generated hands, 12,000 separate validation hands for model selection, and 12,000 held-out test hands. Labels are sampled, so reported prediction errors include label noise. A validation-only accuracy/latency comparison selected eight trees; the deployed model's held-out MAE is 0.01726 and RMSE is 0.02590. Its compressed asset is 5.16 MB. Immutable nested trees reduce lookup overhead, with startup garbage collection outside timed decisions. Policy tuning uses disjoint seeds from final tournament evaluation; synthetic opponent performance is not a prediction of contest rank.

Development logs remain in `artifacts/` and must not be mistaken for final results. The original tuning harness correlated initial seat and button rotation; this limitation is retained in its reproduction path. Final held-out evaluation crosses every starting seat/button pair without changing the frozen policy. An incomplete evaluation using the old seating was discarded. Desktop tests have recorded occasional wall-clock overruns; no hard real-time guarantee is claimed. Platform fallback is check when free, otherwise fold. Agent internal input-error fallbacks are counted separately.

## Public references and provenance

All implementation code was developed independently in this workspace. Public research informed design choices; no external poker implementation or another bot's artifacts were used.

- [Contest rules](https://pokerbot.sequoiacap.com/rules.html), preserved in `RULES.txt`.
- [Poker TDA 2026 rules](https://www.pokertda.com/view-poker-tda-rules/): local standard tournament betting, dead-button and showdown mechanics where the contest text is silent.
- [Brown and Sandholm, Pluribus supplementary material](https://noambrown.com/papers/19-Science-Superhuman_Supp.pdf): supports the importance of action abstraction and bounded computation in multiplayer poker. This bot does not implement Pluribus or its equilibrium training.
- [Gilbert, The Independent Chip Model and Risk Aversion](https://arxiv.org/abs/0911.3100): motivates distinguishing chip expectation from finishing-point expectation. Actual payout input remains necessary.

Training required scikit-learn/NumPy and a C++17 compiler on a fresh, bounded private compute allocation. These dependencies and generated raw training files are not part of the gameplay package. No credentials, SSH control files, control metadata, or remote scripts belong in that package.
