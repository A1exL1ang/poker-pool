# bot_20 â€” final local evaluation

The finished candidate is a dependency-free Python NLHE agent with generated offline equity assets. It passed the mechanics/integration checks below and outscored the matched equity-only control in all nine independently seeded local scenarios. This is evidence against our own test field, not a contest forecast or an equilibrium/optimality claim.

[Runtime ZIP](../artifacts/bot_20.zip) Â· [entry point](../agent.py) Â· [local contract and usage](../README.md) Â· [frozen manifest](../artifacts/final_manifest.json)

## Candidate and selection

The selected model uses 90 features from visible cards and 48 trees per street, trained on 1,000,000 independently generated deals with 512 Monte Carlo trials per deal. A separate 10,000-deal validation set selected the model shape. The 169-class preflop table has 18 range-fraction points, 1â€“7 opponents and 12,000 trials per cell. All deals, ranges, evaluator code and baseline policies were generated or authored within this bot's workspace.

Eight parameter variants were compared in 4,096 development tournaments. The most promising change (less bluffing) then faced the base policy in 4,096 separate validation tournaments. Its paired improvement was âˆ’0.000476 points (standard error 0.004351), so the base settings were retained. No model or policy adjustment was made from final equity/tournament results.

A simulator audit after the first frozen run replaced table identifiers derived from deal seeds with independent opaque identifiers. The bot had used identifiers only for state reset. All 11 runtime/asset files remained byte-identical; the final tournament rerun below used fresh seeds starting at 190,000,000. The first run and scope verification remain in `measurements/pre_opaque_id_*` and `opaque_id_fix_scope.json`.

## Held-out tournament results

Each row contains 512 candidate tables and the same number of matched control tables, with rotating hero seats and independently seeded deals. The control uses the same equity estimator with a simpler pot-odds policy. `robust` includes two copies of this bot; `selfplay` has seven. The other fields contain our own transparent tight, passive, loose, aggressive and equity-based baselines.

**Every payout vector and later blind level below is hypothetical.** Scores are shares of a table's total one-point award, not dollars or a probability of winning. An equal-seat reference is 0.125. Intervals are approximate 95% normal intervals over independent deal-seed blocks; paired trajectories may diverge. Timeout check/fold fallback was enforced for the final runs.

| Field / blinds / payout | Bot score [95% CI] | Control score | Paired advantage [95% CI] |
|---|---:|---:|---:|
| robust / medium / top3 | 0.2280 [0.2117, 0.2442] | 0.0959 | 0.1321 [0.1119, 0.1523] |
| robust / fast / winner | 0.2812 [0.2423, 0.3202] | 0.0762 | 0.2051 [0.1615, 0.2487] |
| robust / slow / flat3 | 0.2416 [0.2287, 0.2544] | 0.0944 | 0.1472 [0.1293, 0.1650] |
| tight / medium / top3 | 0.2802 [0.2633, 0.2970] | 0.0652 | 0.2150 [0.1950, 0.2349] |
| loose / fast / winner | 0.5547 [0.5116, 0.5978] | 0.3418 | 0.2129 [0.1558, 0.2700] |
| passive / slow / rank | 0.2017 [0.1978, 0.2057] | 0.1602 | 0.0415 [0.0356, 0.0474] |
| selfplay / medium / top3 | 0.1342 [0.1186, 0.1497] | 0.0283 | 0.1059 [0.0883, 0.1234] |
| mixed / slow / top3 | 0.3382 [0.3223, 0.3542] | 0.1517 | 0.1866 [0.1651, 0.2080] |
| mixed / fast / flat3 | 0.2812 [0.2708, 0.2917] | 0.1530 | 0.1283 [0.1113, 0.1453] |

Local payout definitions: `top3` = 0.5/0.3/0.2; `winner` = 1/0/â€¦; `flat3` = one-third for each of the first three places; `rank` = 8/36, 7/36, â€¦, 1/36. Exact blind sequences, scores, seeds, actions' timing totals and source hashes are recorded in [final_tournaments.json](../measurements/final_tournaments.json).

## Decision timing and fallbacks

Measurements below are Linux/Python 3.12.7 on our own bounded Engaging allocation. They time `act()` after construction, excluding host observation assembly and JSON decoding. Wall time includes all observed interruptions; no outliers were censored. They do not guarantee the same timing on unknown contest hardware.

| Measurement | Decisions | Median | p99 | Maximum | Over 1 ms |
|---|---:|---:|---:|---:|---:|
| Diverse replay | 64,000 | 63.3 Âµs | 212.3 Âµs | 2854.6 Âµs | 2 |
| Actual candidate tournament actions | 832,247 | not retained | not retained | 23068.1 Âµs | 169 |
| Legal 400â€“700-event history stress | 1,800 | 344.0 Âµs | 511.4 Âµs | 3163.7 Âµs | 1 |

There were **0 decision exceptions, 0 illegal actions, and 169 enforced host fallbacks** in the candidate's final tournament actions. The timeout rate was 0.02031%. The policy's separate internal budget guard used a cheaper legal decision 85 times. Internal guards and host timeouts are distinct counters and need not be disjoint.

Fresh package import plus construction took 1.587 seconds; its first sampled decision took 151.9 Âµs. Model parsing/compilation must occur before the timed interval. If the official runner charges startup to the 1 ms budget, its initialization/adapter arrangement must change. Earlier Windows measurements are retained as development evidence; the final selected build was not bulk-tested on the laptop after the user's quiet-operation request.

[Replay details](../measurements/latency_final.json) Â· [stress details](../measurements/latency_stress.json) Â· [package verification](../measurements/package_verification.json)

## Independent equity accuracy

These tests measure approximation of our documented opponent-range family, not actual opponent behavior. Labels are noisy Monte Carlo estimates; independently seeded deals can contain equivalent card states.

| Test | Deals | Trials/deal | MAE | RMSE | p95 absolute error | Maximum error |
|---|---:|---:|---:|---:|---:|---:|
| Postflop, seed 991237 | 12,000 | 2,048 | 0.01659 | 0.02936 | 0.05699 | 0.43160 |
| Preflop, seed 110327 | 3,000 | 4,096 | 0.00598 | 0.00770 | 0.01491 | 0.05672 |

Estimated label-noise RMSE is 0.00690 postflop and 0.00592 preflop. The rare postflop error of 0.432 is a material limitation; high average accuracy does not make individual decisions exact. The range model also uses sequential conditional sampling and a shared range approximation for multiple opponents. Multiway risk/side-pot strategy is heuristic, even though the simulator's settlement is exact.

[Postflop report](../measurements/equity_heldout.json) Â· [preflop report](../measurements/preflop_heldout.json) Â· [range assumptions and reproduction](../training/README.md)

## Mechanics and offline packaging

**66 unit/integration tests passed.** They cover five/seven-card ranks, exact ICM against independent finish-order enumeration, all-in and partial-blind accounting, raise-to semantics, full/cumulative reopening, side pots/refunds, tied pots and odd chips, heads-up transitions, cutoff scoring, legal actions, state reset, opaque table identifiers, invalid assets, and sensitive bubble decisions. The separate exhaustive evaluator check matched the category counts of all 2,598,960 five-card hands; randomized independent references test kicker ordering and seven-card selection.

The runtime ZIP is **1,565,086 bytes** (below the conservative 100,000,000-byte cap). A fresh isolated Python process loaded both trained assets and returned a legal decision with network functions disabled and built-in file opening disabled during the action. Gameplay has no third-party dependencies or online calls. All collected frozen files and the ZIP hash were checked locally using lightweight reads after transfer.

ZIP SHA256: `0437c96943101baec3eb1b600c59f339dd02db63b5db25218ef49aae48de386e`

Source/asset manifest fingerprint: `b0510189b832d3c2125f713e6ac989666cf5ac5a710010ce3c8f5f7a31b638b9`

Collection verification: 41 frozen files, 0 mismatches, archive hash matched = True.

## Integration limits and completion

The official Agent interface, payouts, later blind schedule, runtime version and initialization accounting were unavailable. The local simulator assumes no antes, moving/live multiway buttons, standard heads-up blind transition, and starting-stack ordering for simultaneous eliminations with equal-stack ties split. An official adapter and tests against the actual host remain necessary. The bot receives only its own hole cards and public information; it does not know opponents' identities or future cards.

All 7 recorded private `bot_20` compute jobs completed; no own allocation remains active. Both authorized helpers finished without requesting allocations. Pre-existing user jobs and the shared SSH connection were not modified. After the quiet-laptop request, substantial computation ran only on our own new bounded Engaging jobs. No push, publication, or contest submission was performed.

The final policy is supported by consistent local held-out performance, strong mechanics coverage, a small offline package and measured timing with honest nonzero tails. Further tuning against these same synthetic opponents would provide diminishing evidence about the unknown contest field; no absolute-optimality claim is made.

## Reproduction

Run substantial commands only in an authorized bounded compute environment while the quiet-laptop restriction applies:

```sh
bash training/run_selected.sh
python -m unittest discover -s tests -v
python training/validate_evaluator.py
python -m benchmarks.freeze
python -m benchmarks.latency --tables 20 --limit 16000 --repeats 4
python -m benchmarks.stress_latency
python -m benchmarks.final_eval --tables 512 --workers 8 --seed-start 190000000
python -m benchmarks.package
```

The separate fresh-label accuracy commands are in `training/README.md`. Reproduction of final seeds is a check of a frozen candidate, not a new tuning set. Full raw development, validation and final measurements remain under `measurements/`; resource provenance remains under `.control/`.
