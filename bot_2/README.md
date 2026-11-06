# bot_2 — independent offline Hold'em bot

One standard no-limit Texas Hold'em bot, implemented in **Python 3.10+ with no runtime dependencies** beyond the standard library. No gameplay network calls, native extensions, or subprocesses. All code and model assets were independently built in this workspace.

The complete supplied `RULES.txt` was read. **The official Agent API, payouts, full blind schedule and timing harness were not supplied.** This implementation defines a local contract; official compatibility requires an adapter. Nothing was submitted, published, committed or pushed.

## Run and test

Run from this directory in PowerShell:

```powershell
Get-Content examples/observation.jsonl | python bot.py
python -m unittest -v tests.test_bot
python -m tools.exhaustive_evaluator
python -m tools.benchmark latency --iterations 50000 --output benchmarks/release_latency.json
```

The sample returns `{"action":"raise","amount":50}`. The JSONL process reads one observation and writes one action per line; stderr contains diagnostics. For the timed host, preload imports/assets and call `Agent.act(Observation)` directly. Keep one instance per table; a new `table_id` or `reset()` clears all learned seat information.

The precise contract is in [INTERFACE.md](INTERFACE.md) and `pokerbot/model.py`. Key assumptions: integer chips; clockwise numeric seats; a live button; pot includes street bets exactly once; call cost is stack-capped; raises specify the **total street contribution**; the host supplies legal bounds and reopening status. Actual payouts enable ICM; omitting them selects chip-EV behavior. Unknown official payouts/blind increments are never invented.

## Design

Position-aware opening/isolation ranges, simulated preflop range equities, short-stack shove EV, selective three-bets, realization-aware calls, postflop value/protection bets, semi-bluffs and pot control. Anonymous opponent statistics use conservative priors and count only observed opportunities. Exact ICM curvature adjusts risk, including an extreme-payout-bubble commitment guard.

Postflop equity uses a locally trained 160-tree, depth-six model over 34 poker features. It predicts showdown shares for multiple opponent counts. Inference is generated ordinary Python; narrow river ranges are enumerated exactly within the assumed range model. The policy is heuristic, not a solved equilibrium or full game-tree search.

## Measured evidence

Only `benchmarks/release_*.json` describes the final gameplay and corrected simulator. Those reports record source hashes. Earlier versions are retained in `benchmarks/development/`.

- **24/24 tests pass**, including 7,500 independent random 5/6/7-card comparisons, independent ICM checks, compiled-model parity, legality, all-ins, state reset, short blinds, side pots, chip conservation and timeout fallback.
- Exhaustive evaluation passed all **2,598,960** five-card hands, their category histogram and **7,462** distinct scores.
- The final equity check passed 100 cross-language feature audits. On 1,500 independent synthetic states against 2,048-sample references: **MAE 0.0288; RMSE 0.0447**. Rare states can have substantially larger errors.
- On Engaging Linux/Python 3.12, 50,000 source-matched calls measured **median 131 microseconds; p99 154 microseconds; maximum 560 microseconds; zero calls over 1 ms**. All 24 tests also passed there. JSON parse/validate/decide/encode p99 was 218 microseconds. This is a measured host result, not a timing guarantee for the official sandbox.
- Final Windows timing over 50,000 calls: **median 215 microseconds; p99 624 microseconds**. There were **175 calls over 1 ms**, with a 12.99 ms maximum. JSON parse/validate/decide/encode p99 was 656 microseconds, excluding process transport. Imports/model load are excluded. Windows thread-CPU accounting is too coarsely quantized to attribute individual outliers.
- Final local tournaments enforce the **1,000-microsecond fold/check fallback**: 3,600 tables across three experimental payout structures, **3,367,811 decisions and zero illegal-action errors**. There were 247 deadline fallbacks overall, 188 by the tested hero. In 250 paired-seed tables per field under the experimental top-three payout, point improvements over the local TAG control averaged **+6.70 / +7.66 / +10.33 / +5.11** against TAG / loose / station / mixed fields. Each paired 95% interval excluded zero. Ordinal and winner-take-all checks also improved on those simple controls. These are local validation results, not contest-strength estimates.

Linux evidence is in `benchmarks/linux_latency.json` and `linux_tests.txt`. The official hardware and timing contract still need verification. **Hard real-time compliance is not guaranteed.**

## Reproduce offline assets and matches

No asset rebuild is needed to run the bot. To reproduce with a C++17 compiler:

```powershell
g++ -O3 -std=c++17 tools/generate_tables.cpp -o tools/generate_tables.exe
./tools/generate_tables.exe 40000
g++ -O3 -std=c++17 tools/train_equity.cpp -o tools/train_equity.exe
./tools/train_equity.exe 240000 16000 160 2> benchmarks/equity_training.log
python tools/compile_equity.py
python -m tools.validate_equity --states 1500 --output benchmarks/equity_validation_final.json
python -m tools.benchmark matches --tables 250 --suites tag,loose,station,mixed --seed-start 180000 --payout top3 --deadline-us 1000 --output benchmarks/release_top3.json
python -m tools.benchmark matches --tables 200 --suites tag,mixed --seed-start 190000 --payout ordinal --deadline-us 1000 --output benchmarks/release_ordinal.json
python -m tools.benchmark matches --tables 200 --suites tag,mixed --seed-start 200000 --payout winner --deadline-us 1000 --output benchmarks/release_winner.json
python tools/package_bot.py
```

Preflop seed `0xB0722026`: 40,000 trials per cell. Equity seed `0xB072E011`: 240,000 training / 16,000 validation states. `assets/equity_audit.jsonl` retains raw state/feature/label examples. Regenerating the model requires recompiling generated Python; startup checks its checksum. Floating-point output may vary slightly across compilers.

## Limits and local rules

The independently authored simulator uses experimental big blinds `[20,30,50,80,120,200,320,500]`, advancing every 25 hands. Experimental payouts are top-three `[50,30,20,0,0,0,0,0]`, ordinal and winner-take-all. The supplied contest rules establish the initial 10/20 blinds, 2,000-chip stacks, eight seats, 25-hand intervals and 200-hand cap; they do not establish these later amounts. Betting edge cases were checked against [Poker TDA rules 45 and 49](https://www.pokertda.com/view-poker-tda-rules/); official competition semantics remain to be confirmed.

Material limitations: synthetic opponent ranges, approximate multiway card removal, noisy training labels, equity abstraction error, fixed bet sizes, simplified side-pot strategy, heuristic use of pairwise ICM, and no policy specifically optimized for the 200-hand boundary. `hands_remaining` is accepted but unused. Opponent history is necessary for useful adaptation. Simple local test opponents do not establish performance against strong contest bots.

`dist/bot_2_runtime.zip` is a small local runtime bundle; `dist/bot_2_complete.zip` also includes tests, training source and benchmark evidence. Neither is an official submission format. Linux validation used only our new preemptible job **28063677** (1 CPU, 512 MB, three-minute cap); it completed in 11 seconds with exit code 0. No bot_2 jobs remain running. No existing user allocations were used or altered, and no credentials were read.
