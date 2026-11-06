# bot_5 — independent offline Hold'em bot

This is one original Python bot, its locally generated equity assets, and a local test environment. **No official Agent interface or starter kit was supplied. This implementation is not yet an official contest adapter.** Nothing has been submitted, committed, pushed, or published.

The full supplied `RULES.txt` was read. It specifies Python, isolated offline play, a 1 ms decision limit, eight-seat sit-and-gos, 2,000 starting chips, starting blinds 10/20, levels every 25 hands, and a 200-hand cap. The actual interface, point vector, and later blind levels remain unspecified. Standard no-limit Texas Hold'em is the user's instructed variant.

## Run and validate

Requires Python **3.10+**, standard library only. Run from this workspace:

```powershell
python -c "import json; from agent import Agent; print(Agent().act(json.load(open('examples/observation.json'))))"
Get-Content examples/observation.json | python agent.py
python -m unittest discover -s tests -v
python tools/exhaustive_evaluator.py
python tools/benchmark.py --tables 48 --repeats 2 --save-corpus
python simulator.py --tables 128 --seed 91000 --lineup mixed
python tools/tournament_suite.py
python tools/package.py
```

On Unix, the JSON-lines command is `python agent.py < examples/observation.json`. Instantiate `Agent` once per table **outside decision timing**, then call `act(observation)`. `reset()` clears table-local learning. A changed `table_id` also resets it. Initialization loads the local assets; no network, external service, model API, or native executable runs in the gameplay path.

## Explicit local interface

`from agent import Agent`; `Agent(seed=5).act(observation)` returns a dictionary containing `action` and integer `amount`.

| Observation field | Meaning |
|---|---|
| `table_id`, `hand_id` | Stable table identifier and changing hand identifier. |
| `hero`, `button` | Indices into `players`; seats run clockwise. Keep seat identities stable within a table. |
| `hole_cards`, `board` | Two private cards and 0/3/4/5 public cards. Integers `(rank-2)*4+suit`, suits `cdhs`, or strings such as `As`, `Td`. No duplicate cards. |
| `players` | 2–8 public records with `stack` (chips behind), `bet` (this street), `committed` (entire hand including blinds/antes), and `folded`. Keep eliminated seats, with zero chips and `folded=true`. Never supply opponents' cards. |
| `pot` | All chips already committed, including the current street, **before hero's action**. |
| `big_blind` | Current big blind. All amounts are integer chips. |
| `legal_actions` | Authoritative mapping described below; the caller must calculate reopening rights and bounds correctly. |
| `history` | Complete append-only prefix for this hand: `{player, street, action, amount}`. Street names: `preflop/flop/turn/river`. Exclude forced blinds from voluntary actions. History is needed for reliable range inference. |
| `payouts` (optional) | Nonnegative, descending points for **remaining finishing places**, first through last. No payout vector is invented when absent. |

Legal actions example: `{"fold":true,"call":10,"raise":{"min":40,"max":2000}}`. When checking is legal, use `{"check":true,...}` and omit `call`. Omit `raise` when betting cannot reopen. A short all-in raise uses equal `min` and `max`. `raise` also represents an opening postflop bet.

**Call amounts are incremental chips; raise amounts are total street contributions (“raise to”).** Checks/folds return amount zero. For example, with 10 already posted and a returned raise to 60, pay 50 more. `current_bet` and `street` are accepted for clarity, but decisions derive the street from board length and take legal bounds from `legal_actions`.

The bot clamps proposed raises to supplied legal bounds, checks instead of folding a free action, and excludes inaccessible side-pot money from call odds. Invalid observations take a check/fold fallback where possible; `last_error` records the reason. `last_info` exposes compact decision diagnostics and does not print to stdout. Only the JSON-lines adapter writes stdout.

## Strategy and assets

- Position- and stack-aware opening, isolation, three-betting, short-stack pushes, and pot-odds-based defense. Suitedness and playability adjust raw preflop equity ordering.
- Preflop equity versus six range widths: 3%, 7%, 15%, 35%, 65%, 100%. The two tight ranges use 100,000 trials/matchup; the other ranges use 30,000.
- **All 1,286,792 canonical flop states**, four range equities each, with 1,024 independent rollouts per range. Suit canonicalization makes lookups cheap. The table is about 10.3 MB.
- Turn/river equity uses bounded samples (40/80), shared-board evaluation, card removal, deterministic common random numbers, and betting-conditioned opponent weights. Per-hand caching avoids repeated work.
- Conservative multiway equity approximation, selective draw bluffs and continuation bets, value sizing, table-local opponent statistics, and exact handling of universally unbeatable shared river boards.
- When points are supplied, an exact subset-DP ICM calculator feeds a pairwise risk-premium heuristic. This is **not** a full multiway ICM action solver.

Range percentiles are cuts of raw heads-up preflop equity, an explicit approximation rather than solved opening or continuation ranges.

Assets and evaluator are original to this workspace. `assets/manifest.json` records hashes and generation parameters. Rebuilding requires a C++17 compiler **only offline**:

```powershell
python tools/build_rank_table.py
g++ -O3 -std=c++17 -o tools/build_equity.exe tools/build_equity.cpp
./tools/build_equity.exe preflop 30000
./tools/build_equity.exe tight 100000
./tools/build_equity.exe flop 4096 0 assets/flop.bin
./tools/build_equity.exe validate 20000 400
```

The flop build took about 656 seconds locally. Rebuild preflop before flop because the range ordering must match. The binary generator targets little-endian machines; the Python loader handles integer-array byte order.

## Evidence and limits

`evidence/` contains raw machine-readable measurements, test logs, individual tournament results, and a saved legal-state latency corpus. See `evidence/RESULTS.md` for the final measured summary. Tests include an independent five-card reference, 5,000 sampled 5/6/7-card comparisons, 3,000 shared-board evaluator checks, suit invariance and flop lookup coverage, ICM permutation checks, legal-action fuzzing, chip conservation, short all-ins, and side pots. An exhaustive **2,598,960-hand five-card census** matches every category count.

Fresh Monte Carlo validation of 400 flop states × four ranges found mean absolute equity error **1.05 percentage points**, p95 **2.69 points**, maximum **5.19 points**. The reference uses 20,000 trials/range and is itself an estimate.

The simulator's defaults of blinds `20,40,80,160,320,640,1280,2560` and points `0.5,0.3,0.2,0,0,0,0,0` are **synthetic test choices, not official values**. Baselines are simple original tight, loose-passive, aggressive, and random policies. They validate functionality and gross strategy, not contest win probability; self-play is included as a symmetry check. Equal stacks eliminated together share local finishing points, an explicit simulator convention.

Remaining limitations: no official adapter; unknown contest hardware/startup accounting, payouts and later blind schedule; approximate opponent ranges and multiway equity; Monte Carlo noise; heuristic postflop pressure and ICM adjustments; no equilibrium-solving claim. Raw wall-clock latency has occasional outliers above 1 ms, recorded rather than hidden. The actual competition sandbox must be timed before submission. The runtime package is well below 100 MB and excludes all development executables and test artifacts.
