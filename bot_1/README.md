# bot_1 — independent offline NLHE agent

One original Python poker bot, with offline equity assets, Windows/Linux native helpers, a Python fallback, and a local validation engine. The full `RULES.txt` was read. No other bots or repositories were inspected. Nothing was submitted, published, committed or pushed.

The official starter kit, interface, payout vector and full blind schedule were not supplied. This is a working local implementation; **official compatibility is unverified**. The exact observation/action assumptions are in [INTERFACE.md](INTERFACE.md).

## Run

Python 3.10+; no third-party Python packages. From this workspace:

```powershell
python run_bot.py
```

Send one JSON observation per line; receive one JSON action per line. Keep the process alive. See [examples/observation.json](examples/observation.json) for a complete observation, or run `python tools/smoke.py`.

Direct integration:

```python
from agent import Agent
bot = Agent(seed=45169)  # initialize once, outside the decision timer
action = bot.act(observation)
```

Create/reset an Agent per table, or change `table_id`. Calls return additional chips; raises return **total street commitment**. Card integers use this project's encoding. [The interface contract](INTERFACE.md) explains cards, bets, pot accounting, legal bounds, history and payouts. Assets/imports/JSON transport are outside measured decision time; the official timing boundary is unknown.

## Strategy

- All 169 preflop classes, six opponent ranges and 1–7 random opponents, generated offline with 40,000 Monte Carlo samples per cell.
- Position/depth-aware opening, price-aware defense, value three-bets, occasional blocker bluffs, and short-stack shove EV with approximate multiple callers.
- Exact 5–7 card evaluation; bounded postflop sampling with known-card removal and bettor strength/draw conditioning. Native heads-up river equity enumerates the inferred range exactly.
- Table-local VPIP/raise/fold estimates, texture/draw-aware betting, and separate eligible-opponent equity for known closed side pots.
- Actual supplied payouts activate exact ICM for terminal heads-up calls and blind-vs-blind shove utilities. Otherwise decisions use chip EV.

This is a heuristic policy, not a solved equilibrium. Multiway ICM, future betting/callers, inferred ranges, equity realization and tie utility remain approximations. No online calls, model inference, compilation or gameplay file I/O occur in `act()`.

## Native helper and portability

Both `assets/fast_equity.dll` (Windows x64, GCC 14.1) and `assets/fast_equity.so` (Linux x64, GCC 12.2) are included. They select automatically at import. The source is `pokerbot/fast_equity.cpp`. Windows Python 3.11.9 and Ubuntu/WSL Python 3.12.3 passed the native suite. Other architectures, loaders and official sandbox restrictions need verification.

Rebuild for a target with GCC/Clang:

```powershell
python tools/build_native.py
```

The Linux build uses `-O3 -std=c++17 -shared -fPIC`; macOS produces a dylib and is untested. The shipped Linux build used compatible C++11 syntax and static GCC/C++ runtimes. Stop your own running processes before rebuilding a Windows DLL. If loading fails, the bot falls back to Python with fewer samples. Force that path before startup with `$env:POKERBOT_PURE_PYTHON='1'`; clear it with `Remove-Item Env:POKERBOT_PURE_PYTHON`.

## Reproduce validation

```powershell
python tools/validate.py
python tools/smoke.py
python tools/exhaustive_evaluator.py
python tools/benchmark.py --iterations 50000
python tools/equity_accuracy.py --positions 480 --reference-trials 20000
python tools/fixture_suite.py --mixed 1200 --each 300 --selfplay 200 --seed 20261108
python tools/final_audit.py
python tools/package.py
python tools/package_smoke.py
```

Regenerate the preflop asset offline:

```powershell
g++ -O3 -std=c++17 -static tools/generate_tables.cpp -o tools/generate_tables.exe
.\tools\generate_tables.exe 40000 assets/preflop.json
```

On Linux omit `-static` if unavailable and invoke `./tools/generate_tables`. Rebuilding assets is never part of gameplay.

## Measured evidence and limits

Machine-readable results and logs are in `benchmarks/`, with source/binary hashes in final timing results. Earlier named experiments are retained.

- **Correctness:** 39 tests pass on Windows and Linux with native helpers. The fallback passes 34 applicable tests on both platforms; five native-only checks skip. Tests cover an independent evaluator, 3,000 legal-action fuzz cases, native/Python parity, ICM, reset/events, side pots, odd chips, refunds and short-all-in reopening. All **2,598,960** five-card hands match expected category counts.
- **Sampling:** 480 positions compared with 20,000-trial same-model references: mean absolute equity error **2.41 percentage points**. This does not validate inferred ranges against real opponents.
- **Latency:** expanded 50,000-decision native stress runs: Windows mean **138 µs**, p99 **533 µs**, **162 over 1 ms**; local Linux mean **128 µs**, p99 **554 µs**, **226 over 1 ms**. Windows fallback: mean **340 µs**, p99 **1,090 µs**, **247/20,000 over 1 ms**. Timers include the full warm `act`, model updates, table resets, side pots and optional ICM, with GC enabled. Outliers are retained. **No hard one-millisecond guarantee is established.** Startup/imports/transport and official hardware remain unverified.
- **Local play:** final 3,200-table suite, zero illegal decisions. Across 1,200 mixed-control tables: **32.108 ± 0.594 SE** synthetic points versus 12.5 equal share. Across 200 self-play tables: **13.55 ± 1.35 SE**. The simple original controls are weak; these numbers cannot estimate official win probability.

The local SNG fixture uses 2,000 chips, eight seats and 200 hands, plus **synthetic** blinds `[20,40,80,120,200,400,800,1600]` every 25 hands and **synthetic** payouts `[50,30,20,0,0,0,0,0]`. Later blinds and payouts are not official facts. The simulator is original test infrastructure, not the official engine.

The Linux helper was built by one new, private Engaging preemptible job (`28063667`, one CPU/1 GB, three-minute limit). It completed successfully in one second; no task jobs remain running and no existing user allocations were touched. Source, logs and the downloaded helper hash are recorded. Modal was not installed. No third-party poker implementation or trained model is bundled.

A private source/assets/tests/evidence bundle is produced at `dist/bot_1.zip`, with `dist/MANIFEST.json`. It is not a competition submission.
