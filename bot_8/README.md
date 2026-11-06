# Kestrel-8

One independently built, offline Python no-limit Texas Hold'em bot. Runtime code is in `agent.py` and `kestrel/`; the original local simulator and reference opponents are in `tools/`. Nothing has been submitted, published, committed, or pushed.

**This is a working local implementation, not an official competition integration.** The complete supplied `RULES.txt` was read. No official Agent interface, point distribution, or blind levels after 10/20 were available. The user's brief specifies standard NLHE. An adapter to the actual event interface is still required.

## Run

Python **3.10 or newer**, standard library only. No installation, API keys, native extension, or network connection is needed for gameplay. Keep `kestrel/preflop.json` beside its Python modules; it loads once at import.

From this workspace in PowerShell:

```powershell
Get-Content examples/observations.jsonl | python agent.py --jsonl
python -m unittest discover -s tests -v
python -m tools.verify_evaluator
python -m tools.stress --states 20000
python -m tools.benchmark --tables 40 --repeat 3 --output evidence/latency_recheck.json
```

On a POSIX shell, the JSON-lines example is `python agent.py --jsonl < examples/observations.jsonl`.

For an in-process host:

```python
from agent import Agent, Observation

bot = Agent()                         # one instance per table
observation = Observation.from_dict(canonical_public_state)
action = bot.act(observation)          # Action(action, amount)
wire_action = action.as_dict()
bot.reset(new_table_id)               # mandatory when reusing an instance
```

`act()` also accepts the canonical dictionary directly. It validates dictionaries; preconstructed `Observation` instances are trusted for speed. Call `validate()` at the adapter boundary. The official adapter must translate both observations and the returned action. JSON transport and process startup are not part of the measured in-process decision time.

## Explicit local contract

The schema is defined in `kestrel/model.py`, with complete examples in `examples/observations.jsonl`.

| Field | Meaning |
|---|---|
| `hand_id`, `table_id` | Hand identifier and table identity. History resets per hand; learned statistics reset per table. |
| `hero`, `button` | Integer seat IDs. Sorted seat IDs proceed clockwise. This contract uses a live button present in `players`. |
| `hole`, `board` | Exactly 2 private cards; 0, 3, 4, or 5 public cards. No opponent hole cards. |
| Cards | Integer `4*(rank-2)+suit`, with suits `c,d,h,s = 0,1,2,3`; dictionaries also accept strings such as `As`, `Td`, `10h`. |
| `players` | All players who started this hand, including folded and all-in players, excluding players eliminated in earlier hands. |
| Player `stack` | Chips remaining **behind**; excludes all committed chips. |
| Player `committed`, `invested` | Total commitment on this street, and total investment in this entire hand, respectively. |
| Player `folded`, `all_in` | Public eligibility flags. Hero must be eligible to act. |
| `pot` | All contributed chips, **including current-street bets**; must equal the sum of `invested`. |
| `to_call` | Additional chips needed to match the bet; may exceed hero's stack. A call automatically pays the smaller amount. |
| `min_raise_to`, `max_raise_to` | Minimum and maximum **total street commitment**, not additional chips. A short all-in below the minimum is legal only at hero's actual stack cap. |
| `can_raise` | Host-supplied raising/reopening permission. The bot never overrides this restriction. |
| `big_blind` | Current big blind; no hardcoded schedule in the bot. |
| `history` | Complete chronological current-hand public actions with strictly increasing IDs. No blind-post events. Actions are `fold`, `check`, `call`, `raise`. |
| Event `amount` | New total street commitment for a bet/raise, chips actually paid for a call, zero otherwise. `pot_before` and `to_call` describe the state before that event. |
| `payouts` | Optional descending finishing-place **points**, including zeros. Supply remaining-place points for the players still in the tournament. If absent, use chip EV. These are not the event's dollar prizes. |
| `hands_remaining` | Optional metadata; currently does not change the policy. |

Returned actions are `fold`, `check`, `call`, or `raise`. `amount` is zero except for `raise`, where it is the **raise-to** amount. Opening bets also use `raise`. There is no separate `all_in` action. A central safety gate caps raises, respects reopening, and converts a free fold to a check. Unsupported/malformed JSON is reported as an explicit error by the CLI, rather than silently fabricating a game state.

The host adapter must preserve blind positions and betting order. Empty/dead buttons and a different official seat model require explicit translation; they are not silently guessed.

## Strategy and cost controls

- Position- and effective-stack-dependent opening, isolation, value reraising, and limited blocker bluffs. Short stacks use a fold-equity/all-in approximation.
- Original offline preflop Monte Carlo tables for all 169 starting-hand classes against eight strength ranges. Each range cell uses 40,000 trials; range boundaries include whole hand classes to preserve suit symmetry.
- Exact 5–7-card ranking. Postflop equity jointly samples all active opponents without card collisions, weights their ranges using observed betting, and accounts for side-pot eligibility and ties.
- Value betting, smaller continuation bets on suitable boards, draw semibluffs, and conservative calling with equity-realization and sampling-error adjustments.
- Bayesian-smoothed per-seat statistics based only on observed public actions. Preflop folds and postflop folds are tracked separately. No team identification or cross-table learning.
- Optional exact ICM thresholds for expensive, eligible heads-up calls; cheaper risk adjustments elsewhere. Winner-only points reduce to chip EV. No payout schedule is invented inside the policy.

Default `Agent(sample_scale=0.5, time_budget_us=450)` caps nominal sampling at 32 heads-up flop/turn samples and 40 river samples, with fewer samples multiway. Sampling stops early when the 450-microsecond work target is reached. The target reserves time for decision bookkeeping and return. It is **not a hard real-time guarantee**: OS scheduling, garbage collection, parsing, and the host's hardware can still cause timeouts.

For exactly reproducible offline decisions, use `Agent(time_budget_us=None)`. Cards, mixed actions, and sample sequences use deterministic hashes of the observation; enabling the time guard can change the number of samples under different machine load. Larger `sample_scale` is for offline experiments, not a promise of meeting the contest limit.

## Measured evidence

All measurements are local experiments against original simple reference heuristics. They do not establish contest rank or equilibrium strength. Initial exploratory results are retained separately; the following holdouts use the final gameplay implementation and fresh seeds.

**Correctness:** 24 unit tests passed, including 3,600 randomized 5/6/7-card comparisons against an independent five-card oracle; exact ICM checked against permutation enumeration; side-pot payouts; short-raise reopening; heads-up order; state reset; and legal-action conversion. All **2,598,960** five-card hands match the known category counts. A further **20,000** randomized valid-contract states produced no exceptions or illegal actions. Reachable tournament states also enforce legality and chip/point conservation on every hand.

**Windows 11 / Python 3.11.9 latency:** 6,781 observations from 40 local tournaments, replayed three times through the typed interface. Includes observation processing, range inference, policy, and action safety gate. Excludes imports, initialization, JSON decoding/encoding, and IPC.

| Interface | Calls | Mean | p95 | p99 | Calls over 1 ms |
|---|---:|---:|---:|---:|---:|
| Typed `Observation` | 20,343 | 238.4 us | 518.5 us | 767.5 us | 104 (0.51%) |
| Dictionary, including validation | 6,781 | 285.6 us | 521.6 us | 996.8 us | 66 (0.97%) |

Maximum observed typed wall time was 10.53 ms. Windows process-CPU timer resolution was too coarse for per-call percentiles; use the reported wall measurements. **Strict official one-millisecond compliance has not been established.** The deadline fallback was therefore also exercised in tournament evaluation.

**Linux / Python 3.12.7 cross-check:** one newly allocated CPU on MIT Engaging, with the same final runtime source hashes. Job `28063676` requested 1 GB and a four-minute limit on `mit_preemptable`, completed successfully in 10 seconds, and is no longer running. All 24 unit tests and the exhaustive evaluator check also passed there.

| Linux interface | Calls | Mean | p95 | p99 | Maximum | Calls over 1 ms |
|---|---:|---:|---:|---:|---:|---:|
| Typed `Observation` | 21,393 | 106.7 us | 271.5 us | 294.3 us | 721.5 us | 0 |
| Dictionary, including validation | 7,131 | 126.7 us | 294.6 us | 320.4 us | 467.9 us | 0 |

This provides measured headroom on that Linux allocation, not a guarantee about the unpublished official host. Only the bot's private remote directory and its own new job were used; no existing user allocations were used or modified.

| Local holdout | Tables | Kestrel mean points | Paired control | 95% CI for paired difference |
|---|---:|---:|---:|---:|
| Mixed styles, laboratory points, 1 ms fallback enabled | 1,200 | 5.630 | 2.418 | +3.006 to +3.418 |
| Aggressive styles, winner-only points | 400 | 0.452 | 0.130 | +0.268 to +0.376 |
| Tight styles, equal points for top three | 400 | 0.747 | 0.367 | +0.322 to +0.438 |

The mixed holdout made 220,367 hero decisions, of which 737 (0.33%) used the local timeout fold/check fallback. It played 172,136 table hands and checked 1,013,110 total-player actions. Hero seats rotate; paired controls use the same seed and the original reference policy at that seat. Confidence intervals are approximate normal intervals over table-level paired differences; changing actions changes later dealt-card paths as players bust, so these are not duplicate-board tournaments.

The **laboratory-only** point schedule is `[8,5,3,2,1,0,0,0]`. Experimental blind levels are `(10/20,15/30,25/50,40/80,60/120,100/200,150/300,250/500)`, advancing every 25 hands. Only the first level, interval, starting stack 2,000, eight seats, and 200-hand cap come from the supplied competition rules. Later levels and point schedules above are test inputs, not claims about the event.

An exact river-range audit over 1,000 states found 3.88 percentage points mean absolute sampling error, 6.20 points RMSE, and 13.97 points p95 absolute error; worst error was 29.97 points. This compares 40 samples with exact enumeration under the **same heuristic opponent model**, not with real opponent strategy. A time-limited call may use fewer samples.

Raw results: `evidence/tests.txt`, `evidence/evaluator_exhaustive.json`, `evidence/stress.json`, `evidence/equity_audit.json`, `evidence/latency_final.json`, and the three `evidence/holdout_*.json` files. Source-matched Linux validation is recorded separately in `evidence/linux_latency.json`, `linux_tests.txt`, `linux_evaluator.json`, `linux_source_hashes.json`, and `linux_job_status.txt`.

## Reproduce the experiments

```powershell
python -m tools.evaluate --tables 1200 --seed 182000 --field mixed --control --enforce-deadline --output evidence/holdout_mixed_recheck.json
python -m tools.evaluate --tables 400 --seed 222000 --field aggressive --schedule winner --control --output evidence/holdout_aggressive_recheck.json
python -m tools.evaluate --tables 400 --seed 232000 --field tight --schedule flat --control --output evidence/holdout_flat_recheck.json
python -m tools.audit_equity --states 1000
python -m tools.package
```

Use `--budget-us 0` in `tools.evaluate` or `tools.benchmark` to disable the adaptive timer for deterministic studies. Adaptive-timer and enforced-timeout results can vary with host load. `--field abc`, `passive`, and other documented argparse options are available for further local experiments.

To regenerate the original preflop asset with a local C++17 compiler:

```powershell
g++ -O3 -std=c++17 tools/precompute.cpp -o tools/precompute.exe
./tools/precompute.exe 40000 | Set-Content -Encoding utf8NoBOM kestrel/preflop.json
```

The generator uses a fixed xorshift seed recorded in the asset. C++ is only an offline generation tool; the bot needs neither a compiler nor the generated executable.

## Package, boundaries, and remaining work

`python -m tools.package` builds `dist/kestrel8.zip`, containing this single bot's runtime sources, preflop asset, and README. **Extract it before importing or running.** `dist/manifest.json` records byte counts and SHA-256 hashes. The separate private validation archive includes the local tests/tools; it is not a second bot entry. No credentials, `.control` files, repository metadata, logs from other agents, or external code are packaged.

The remaining integration work is to map the real Agent API, confirm initialization/time accounting, provide the actual point schedule and blind observations, and rerun the tests and timing on the official host. The simulator uses a moving live button, no antes/rake, a conventional starting-stack rule for simultaneous elimination ties, and its own local action protocol. It is a validation harness, not a certified reproduction of the tournament platform.

The strategy is heuristic and exploitable; it is not a solved NLHE equilibrium or a guarantee of winning. Main limitations are finite-sample equity error, coarse opponent ranges, limited public-history observations after hero folds, approximate future-street realization/fold equity, approximate multiway ICM risk, and no special optimization for the final hand or score weights across T1–T6. Python/hardware latency remains an integration risk despite the budget guard.

Competition source: [supplied official rules](https://pokerbot.sequoiacap.com/rules.html), also preserved in full as `RULES.txt`. The local raise/reopening and short-all-in behavior was cross-checked against the [Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/) and its [short opening all-in clarification](https://www.pokertda.com/forum/index.php?topic=1456.0). These public rules are reference material; event-specific platform rules take precedence.
