# bot_6 — independent offline Hold'em agent

An original Python no-limit Texas Hold'em bot, built from the supplied `RULES.txt`. No other entries, starter kits, or opponent code were used. The gameplay path uses only Python's standard library and a locally generated equity asset; it makes no network or AI calls.

**This is a working local implementation, not an official-interface-compatible submission.** The official Agent interface, point payouts, and later blind levels were not provided. An adapter and measurements on the actual competition hardware are still required. Nothing has been submitted, committed, or published.

## Run

Python 3.10+; tested on Python 3.11.9. No pip installation needed. From this workspace:

```sh
python agent.py < examples/open.jsonl
```

PowerShell itself does not support `<` input redirection. Use this equivalent there:

```powershell
Get-Content examples/open.jsonl | python agent.py
```

Prefer a persistent imported agent for gameplay:

```python
from agent import Agent

agent = Agent(seed=60062026)
action = agent.act(observation)  # a Python dict, or validated bot6.interface.State
```

Construct/import before timing a decision. Keep one instance per table. `table_id` changes automatically reset opponent statistics; `reset()` is also available. `get_action` is an alias of `act` with the same local schema. The JSON-lines runner outputs one JSON decision per input line; invalid input produces an explicit error object and stderr message. It is a development transport, not a claimed competition protocol.

## Local observation contract

| Field | Meaning |
|---|---|
| `hero`, `button` | Integer seats; increasing seat number is clockwise. |
| `hole_cards`, `board` | Two private cards and 0/3/4/5 public cards. Strings such as `As`, `Td`, or integers `rank_index*4+suit_index`, ranks `23456789TJQKA`, suits `cdhs`. No hidden opponent cards. |
| `players` | All players dealt into this hand, including folded players, excluding previously eliminated seats. Each has `seat`, `stack` (chips behind), `bet` (current street), `contributed` (entire hand), `folded`, `all_in`. Defaults: zero wagers, not folded; a zero stack implies all-in. |
| `big_blind`, `pot` | Current positive BB and total chips already committed, **including current street wagers**. Antes, if any, belong in `contributed` and `pot`. |
| `to_call` | Additional chips needed to call, capped at hero's stack. Optional: derived from current bets. |
| `min_raise_to`, `max_raise_to` | Legal total hero street contribution after raising. Supply these from the engine. The defaults are only a convenience and cannot reconstruct a prior raise size. |
| `legal_actions` | Allowed names from `fold`, `check`, `call`, `raise`. Supply engine legality, including reopening after short all-ins. `can_raise=false` also prohibits raising. |
| `history` | Chronological **current hand only** action dictionaries: `seat`, `street` (`preflop/flop/turn/river`), `action`; optional `amount`, `paid`, `pot_before`. `amount` is a total street contribution. Blind posts should not be labeled calls/raises. |
| `table_id`, `hand_id` | Stable IDs; advance `hand_id` each hand and change `table_id` each table. |
| `payouts` | Optional nonnegative descending point vector for the players still in this hand, including folded players. No payout structure is assumed when omitted. |
| `risk_premium` | Optional external equity premium, 0–0.25, for heuristic decisions. Exact ICM supersedes it in the supported all-in case. Otherwise a small payout-aware heuristic is used when payouts are known. |

Output is `{"action":"fold"}`, `{"action":"check"}`, `{"action":"call"}`, or `{"action":"raise","amount":N}`. **Raise `amount` is raise-to, not an increment.** A raise below the full minimum is only emitted as an all-in when the engine explicitly permits raising. Accurate total contributions are necessary to exclude unwinnable excess wagers and identify side pots; inconsistent totals fall back to a single-pot approximation.

## Strategy

- Position and effective-stack-sensitive opening, isolation, 3-betting, short-stack shoving, and pot-odds defense. Deep opens use a separate playability ordering to favor suited connectors and avoid dominated offsuit hands; shallow shoves use all-in equity ordering.
- Original offline preflop Monte Carlo equities: 169 hand classes × 11 opponent-range widths × 20,000 trials per cell, plus 40,000 trials per class to order ranges. Whole hand classes preserve suit symmetry.
- Exact 5–7 card bitmask evaluation. Postflop samples draw joint opponent hands without card collisions, apply board-dependent action likelihoods, and deal missing community cards. Ties and side-pot shares are fractional.
- Value sizes vary with board texture, opponents, and stack-to-pot ratio. Bluffs favor draws, position, and selected blockers. Conservative realization/uncertainty adjustments discourage marginal calls.
- Table-local, prior-smoothed VPIP/raising statistics influence ranges. Only observed preflop action opportunities enter the denominator, so an unobserved player is not mislabeled as having folded. Only actions observed before the bot's most recent decision are learned; there is no hidden-hand or cross-table information.
- Exact ICM utility is available in `bot6/icm.py`. Heads-up calls against an all-in use exact ICM win-probability thresholds when complete contributions and payouts are supplied. Other situations use a documented risk heuristic. No unknown contest payout or future blind level is invented.

`Agent(sample_budget=120, time_budget_us=500)` controls postflop work. The sample count is bounded, with an internal elapsed-time check every four worlds. Set `time_budget_us=0` only for deterministic offline experiments; a fixed seed alone cannot make deadline-limited sampling independent of machine load. This is a heuristic policy, not a solved equilibrium or trained game-tree search.

## Validate and reproduce

```powershell
python -m unittest discover -s tests -v
python -m tools.exhaustive_evaluator
python -m tools.simulate --tables 300 --capture --enforce-timeout --output benchmarks/mixed.json
python -m tools.benchmark --iterations 30000 --output benchmarks/latency.json
```

The simulator is original validation code with tested side pots, odd chips, short all-in raises, cumulative reopening, heads-up blinds, and uncalled refunds. It checks chip conservation after every hand and rejects illegal actions. Its opponents are deliberately simple original test fixtures. `--field tight`, `calling`, `maniac`, and `self` provide additional stress tests. `self` uses copies of this same bot. `--enforce-timeout` applies the rules' fold/check fallback for locally measured calls over 1 ms; without it timing is recorded but actions are not replaced.

**Simulation assumptions:** 2,000 initial chips, eight seats, 200-hand cap, and starting 10/20 blinds match the supplied rules. Doubling blinds every 25 hands is a **synthetic local schedule**; the rules specify only the interval, not subsequent sizes. Reported win rate/final chip share is a **winner-take-all diagnostic**, not the unknown contest point score. A survivor's chip fraction at the cap equals its winner-take-all ICM equity. Simultaneous busts tied in starting stack receive arbitrary seeded ordering for local rank reporting.

Regenerate the offline equity asset (g++ is needed only for this step):

```powershell
g++ -O3 -std=c++17 tools/build_equity.cpp -o tools/build_equity.exe
./tools/build_equity.exe 20000 assets/preflop.json
```

Evidence is saved in `benchmarks/`. **39 tests pass on both Windows/Python 3.11.9 and Linux/Python 3.12.7.** The evaluator passed an exhaustive census of all **2,598,960 five-card hands**, plus 7,500 independent 5/6/7-card reference comparisons and 500 suit-invariance checks. Linux runtime SHA-256 hashes match the local runtime exactly. `benchmarks/FINAL_RESULTS.json` records source hashes and final results; historical files retain intermediate measurements.

| Warm dictionary decision benchmark | Median | p99 | Maximum | Over 1 ms |
|---|---:|---:|---:|---:|
| Windows, 30,000 calls | 0.306 ms | 0.854 ms | 36.224 ms | 210 |
| Engaging Linux, 30,000 calls | 0.230 ms | 0.517 ms | 2.307 ms | 3 |

Both include card parsing/validation, garbage collection, recorded states, and multiway/ICM stress cases. Windows module import took 124–164 ms in five fresh processes, so initialization must precede gameplay timing.

The final Windows mixed-field test won **446/1,000 tables**, with the 1 ms timeout fallback enforced. Additional Windows tests produced 51.0% wins against tight fixtures, 51.7% against calling fixtures, and 22.7% against aggressive fixtures (300 tables each); self-play won 13/100, consistent with the symmetric 12.5% baseline. Linux replications and capped-table chip-share statistics are retained separately. These fixtures are simple and provide no contest-strength guarantee; time-limited sampling and timeout fallbacks also make seeded results load-dependent.

Build and verify the runtime bundle:

```powershell
python -m tools.summarize
python -m tools.package
python -m tools.validate_package
```

`dist/bot6.zip` contains the gameplay code, preflop asset, example, and README. `dist/manifest.json` contains file hashes. Tests, generation tools, simulator, and full evidence remain in the workspace rather than the small runtime ZIP.

## Limits and integration risks

- The official interface and timing boundary are unknown. Benchmarks cover warm `Agent.act`, including dictionary validation/card parsing; import, JSON transport, and process startup are excluded.
- The one-millisecond rule is **not guaranteed** on this Windows host: scheduling/GC outliers remain even with an internal sampling deadline. Rebenchmark in the official sandbox and lower the time/sample budgets if necessary.
- Small Monte Carlo samples create noisy decisions, especially with many players or heavily weighted ranges. Range shapes, action likelihoods, realization factors, bet sizes, and most tournament adjustments are handcrafted approximations. Bounded rejection has a uniform fallback under extreme card collisions.
- Side-pot equity estimates are chip-weighted; nonterminal betting EV and multiplayer ICM are approximations. Exact ICM call thresholds use fractional equity as an approximation when chops are possible.
- Local fixture wins do not establish strength against other contest entries. Payouts, blind progression, opponent mix, and the 200-hand cap can materially change optimal strategy.
- Assets load during import. Gameplay is CPU-only and offline. The preflop asset was generated locally. Engaging job **28063496**, `bot_6_validate`, used its own new one-CPU/2 GB preemptible allocation, completed successfully in 71 seconds, and left no active job. No existing user allocation was used or altered; no Modal jobs were started.

Source, assets, simulator, tests, generation tools, and benchmark evidence are all retained in this workspace.
