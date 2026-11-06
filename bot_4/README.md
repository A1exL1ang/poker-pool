# bot_4 — independent no-limit Texas Hold'em bot

A self-contained Python 3.11+ tournament policy with offline-trained equity estimates. Runtime dependencies: **Python standard library only**. No network calls, native extensions, subprocesses, external services, or live Monte Carlo in gameplay. One independent bot; the simple policies under `tools/` are local test opponents only.

**Integration status:** the full supplied `RULES.txt` was read. No official Agent API, payout table, or later blind levels were supplied. This project defines and tests its **own interface**, not official compatibility. It has not been submitted, published, committed, or pushed. Wire the actual event's interface and verify its timing harness before use.

## Measured result

The final source-matched validation passed 36 tests, an exhaustive 2,598,960-hand five-card census, all 292,825 three/four-card boards, and 2,300 local tournaments. `evidence/validation_run.json` records commands and runtime source hashes. Fresh exact river calibration has RMS equity errors of 0.0262–0.0397 across the eight outputs.

| Local opponent field | Tables | Mean points ± standard error |
|---|---:|---:|
| Mixed, illustrative top-three payout | 500 | 0.2900 ± 0.0094 |
| Simple equity baseline | 300 | 0.2880 ± 0.0114 |
| Tight baseline | 300 | 0.2474 ± 0.0131 |
| Calling station | 300 | 0.4179 ± 0.0068 |
| Identical-policy self-play | 400 | 0.1249 ± 0.0087 |
| Mixed, winner-take-all | 500 | 0.4316 ± 0.0218 |

Equal-share reference is 0.125 in every row. These simple local fields do not predict contest performance. Windows Python 3.11 dictionary decisions measured **140.3 µs median / 448.0 µs p99**, with **93/40,000 above 1 ms** (maximum 30.67 ms). Linux/WSL Python 3.12 measured **128.7 µs median / 421.1 µs p99**, with **42/40,000 above 1 ms**. Full reports, including typed calls, realistic replay states and initialization costs, are in `evidence/latency_final.json` and `evidence/latency_linux.json`. The hard per-call deadline is therefore **not certified** on this host.

## Run and test

Run these commands from this workspace:

```powershell
python -m unittest discover -s tests -v
python tools/validate.py --full
Get-Content examples/decision.json | python agent.py
python tools/benchmark.py --rounds 5
python tools/calibrate.py --states 800
python tools/simulate.py --tables 500 --seed 81000 --field mixed --capture --output evidence/mixed_final.json
python tools/package.py
```

The JSON-lines CLI accepts one complete observation per line and returns one action per line. It keeps the same `Agent` alive across lines and automatically resets when `table_id` changes. JSON transport, process launch, module import, and model loading must happen outside the per-decision budget. The supplied benchmark times both typed decisions and dictionary conversion plus decisions; it reports initialization separately.

`dist/bot_4.py` is a generated single-file version with the same `Agent` API and JSON-lines CLI; it needs no adjacent assets or package files. Run `Get-Content examples/decision.json | python dist/bot_4.py`. Packaging checks its actions against the modular bot on 500 recorded public observations.

For in-process use:

```python
from agent import Agent, State, validate_state

bot = Agent(seed=404)          # initialize once, outside the decision timer
action_dict = bot.act(observation_dict)
# Or construct a typed State once in your adapter:
state = State.from_dict(observation_dict)
validate_state(state)         # optional integration validation
action = bot.decide(state)    # Action(action='raise', amount=44), for example
bot.reset()                  # mandatory at every new table if table_id reused
```

`observe(state)` is an optional public-history callback, useful at hand end when the bot has folded. It never takes opponent private cards. `decide` also consumes the cumulative current-hand history, deduplicating already-observed events.

## Local observation contract

All chip values are integers. Seat IDs are unique integers in clockwise numeric order and remain stable within a table. Include every seat dealt into the current hand, including folded seats, so the button and position remain meaningful. Omit previously eliminated seats. Never send folded opponents' private cards.

| Field | Meaning |
|---|---|
| `table_id`, `hand_id` | Table identifier and increasing hand identifier. |
| `hero`, `button` | Acting seat and dealer seat. |
| `hole`, `board` | Two private cards; board length 0, 3, 4, or 5. |
| `players` | Records with `seat`, `stack`, `committed`, `contributed`, `folded`, `all_in`. |
| `stack` | Chips behind, after current wagers. |
| `committed` | This street's total contribution, including blinds but excluding antes. |
| `contributed` | Entire hand's contribution, including previous streets and antes. Required for correct short-stack pot eligibility. |
| `pot` | All chips in the middle, including **current** wagers and folded players' money. |
| `big_blind`, `ante` | Current amounts, not an inferred schedule. `ante` defaults to zero. |
| `to_call` | Full additional call cost, before capping at the hero's remaining stack. |
| `min_raise_to` | Minimum **full** raise in total street contribution. |
| `max_raise_to` | Maximum legal total street contribution, normally committed + stack. |
| `can_raise` | Engine-authoritative reopening flag. A short all-in does not necessarily reopen action. |
| `history` | Public events for this hand in chronological order. Blinds/antes are not action events. |
| `payouts` | Optional descending points by finishing rank, starting at first place. Unknown/omitted means chip-EV policy, not an invented payout. |

Each history event has `seat`, `street` (`preflop/flop/turn/river`), `action` (`fold/check/call/raise`), `amount` (total street contribution **after** action), `to_call` (full cost faced **before** action), and `pot_before`. A bet into an unopened pot is `raise`. Integer cards use `rank * 4 + suit`: ranks `23456789TJQKA`, suits `cdhs`. The dictionary adapter also accepts strings such as `As`, `Td`, `2c`.

Actions are `{"action":"fold|check|call|raise","amount":0}`. Only `raise` uses a nonzero amount: it is **raise-to**, never a raise increment or additional payment. Calls are implicitly capped by the player's stack. The legality guard clamps full raises and permits a subminimum raise only when it is the player's own all-in. It respects `can_raise=False` and does not claim to reconstruct official engine legality.

The integration must supply accurate history, betting bounds and current contributions. Malformed observations deliberately raise errors instead of silently guessing an incompatible protocol. The official sandbox's documented error fallback is not implemented as a substitute for integration validation.

## Strategy and assets

- Position-aware opens and isolation raises; range-weighted calls, value three-bets, selective suited-ace bluffs, and stack-aware short-stack shove EV.
- Offline equity estimates for current hand strength, random ranges, preflop top-50/20/5% ranges, and three postflop continuing ranges. These are **heuristic range families**, not a solved equilibrium strategy or exact opponent posterior.
- Texture-sensitive value sizing, selective continuation bets and draw semi-bluffs, pot-odds defense, multiway equity discounting, eligible-side-pot adjustment on short all-in calls, and exact handling of several unbeatable public boards.
- Shrunk within-table VPIP, preflop-raise, aggression and fold statistics. No cross-table identity or memory.
- Optional payout-aware **approximate** risk premiums. `bot/icm.py` also contains exact ICM for local scoring; the gameplay approximation is not an exact ICM action solver.

`tools/equity.cpp` is an original C++17 offline simulator and multi-output gradient-boosted tree trainer. The final training configuration is seed `19471106`, 450,000 states, 768 sampled opponent/runout trials per state, 112 trees of maximum depth 6, and a 10% held-out split. All eight outputs share each tree's splits. That is 345.6 million postflop trials; there is no training during gameplay.

`tools/compile_model.py` turns the trained trees into ordinary Python branches. Eight fixed-point outputs occupy independent 32-bit lanes of a Python integer. One packed addition per tree replaces eight float additions. The generator checks lane bounds; tests compare against the original floating-point traversal with a maximum allowed error of 0.00001. The generated source is an asset, not hand-maintained policy code.

Rebuild all model assets locally with a C++17 compiler:

```powershell
g++ -O3 -std=c++17 tools/equity.cpp -o tools/equity.exe
& tools/equity.exe 450000 768 112 2> evidence/training_final.log
python tools/compile_model.py
python -m unittest discover -s tests -v
```

Training writes `assets/preflop.json`, `assets/equity.json`, and training evidence. Compilation writes `bot/compiled_model.py`. Do not update assets while a new bot process is importing them. Generated data are entirely local simulations from this project's evaluator and random seed. Compiler/STL versions can change exact floating-point results; preserve the shipped assets for byte-for-byte inference reproducibility.

## Validation and limits

Machine-readable evidence and logs are under `evidence/`. `model_training.json` gives held-out errors; `exact_river_calibration.json` checks unseen river states against all 990 legal opponent hole-card combinations. Tests cover an independent five-card reference, randomized 5/6/7-card hands, Python/C++ feature parity, all ranks of four-of-a-kind turn boards, packed-model parity, legal actions, short all-ins and cumulative reopening, side pots, ties/odd chips, reset semantics, ICM and complete local tournaments.

The local simulator uses eight 2,000-chip seats, initial 10/20 blinds, blind changes every 25 hands and a 200-hand cap, as stated in the supplied rules. Its later big blinds `[30,50,80,120,200,300,500]`, zero antes, and test payouts (`[0.5,0.3,0.2,0,...]` or winner-take-all) are **illustrative local choices, not official values**. Simultaneous busts are ordered by start-of-hand stack and equal-stack ties share affected points; this is also a local assumption. Surviving players at the cap receive exact ICM points.

The simulator enforces legality and chip conservation. Its opponents are deliberately simple and do not represent the contest field. Simulation results support regression testing and limited sanity checks, not a forecast of contest placement. Model estimates are approximate, especially for rare textures and ranges outside the training families. Multiway equity and elimination risk use approximations; there is no game-theoretic exploitability guarantee, exhaustive strategy search, or full side-pot-aware multiway utility solver.

Latency varies with CPU, interpreter, garbage collection and scheduling. The benchmark retains GC and reports medians, percentiles, maxima and every wall-clock decision above 1 ms. Submillisecond median/p99 measurements **do not guarantee** every decision will satisfy the hard contest deadline. Startup is separately reported. Re-measure in the actual sandbox.

No remote compute was used. MIT Engaging access became verified after local model training was essentially complete; no allocation was needed. All generated data and benchmarks use this workspace. No credentials, other bots, other projects, remote user allocations, or unrelated jobs were inspected or changed.

`python tools/package.py` creates `dist/bot_4.py`, `dist/bot_4.zip` and a SHA-256 manifest from a fixed allowlist. These are local delivery artifacts, not submissions. The ZIP contains runtime files; the complete training sources, test tools and evidence remain in this workspace.
