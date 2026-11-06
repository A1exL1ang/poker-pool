# Measured results

The model and policy were frozen before these held-out seeds were used.
All final substantial tests ran in a new bounded bot_16 Engaging allocation; the laptop remained quiet.

## Held-out local tournaments

These are independently authored synthetic opponents and synthetic point schedules, not contest results.
Strategy runs were parallel; their latency counts retain scheduling contention. The deadline TAG run was serial.

| Field / test | Tables | Mean points | 95% normal CI | Equal share |
|---|---:|---:|---|---:|
| mixed | 1000 | 33.255 | [32.088, 34.421] | 12.500 |
| tag | 1000 | 30.484 | [29.290, 31.678] | 12.500 |
| aggressive | 1000 | 33.235 | [32.119, 34.351] | 12.500 |
| tight | 500 | 31.268 | [29.580, 32.956] | 12.500 |
| adversarial | 500 | 37.926 | [36.128, 39.724] | 12.500 |
| calling | 500 | 46.111 | [45.123, 47.099] | 12.500 |
| polar | 500 | 39.585 | [38.515, 40.654] | 12.500 |
| selfplay | 400 | 14.431 | [12.632, 16.229] | 12.500 |
| jam | 300 | 35.154 | [34.013, 36.295] | 12.500 |
| winner | 500 | 0.484 | [0.443, 0.524] | 0.125 |
| linear | 500 | 6.681 | [6.528, 6.835] | 4.500 |
| doubling | 500 | 29.460 | [27.674, 31.246] | 12.500 |
| control_mixed | 300 | 13.709 | [11.612, 15.806] | 12.500 |
| control_tag | 300 | 11.171 | [9.246, 13.097] | 12.500 |
| deadline_tag | 600 | 30.374 | [28.852, 31.895] | 12.500 |

## Equity accuracy

Held-out Monte Carlo target MAE: 0.020953; RMSE: 0.032020; p99 absolute error: 0.110348.
Targets describe explicit synthetic ranges. Remaining model/range errors and sequential multiway sampling are limitations, not eliminated by this test.

## Decision timing

Full-act measured calls: 71966; wall-time calls above 1 ms: 6.
All observed outliers are retained. Per-workload wall and CPU distributions, cold first call, import cost, and error counters are in benchmarks/latency_final_linux.json.
Import/model initialization is outside the measured action deadline. The official hardware and timing boundary are unknown; this does not establish official timing compliance.

| Timing workload | Decisions | p99 wall time | Maximum | Over 1 ms |
|---|---:|---:|---:|---:|
| Serial deadline-enforced TAG tournaments | 128,796 | 0.220 ms | 0.399 ms | 0 |
| Saved tournament observations | 50,000 | 0.219 ms | 1.045 ms | 1 |
| Random legal, street-balanced observations | 9,965 | 0.234 ms | 1.119 ms | 2 |
| Eight-survivor ICM flop stress | 1,000 | 0.203 ms | 32.857 ms | 2 |
| Covered preflop ICM stress | 1,000 | 0.093 ms | 1.013 ms | 1 |

The other stress workloads had zero overruns. Legal histories up to 421 entries
were included; their maximum was 0.541 ms. The cold first action took 0.156 ms.
Import took 56.4 ms in the latency process and 439.5 ms in the freshly extracted
offline-package probe. These initialization costs are explicitly outside the
action timer and would need reconsideration if the official interface times
startup.

The six replay/stress spikes also appear in process/thread CPU time, so they
cannot honestly be attributed solely to OS scheduling. Their cause was not
isolated. The bot therefore does **not** claim that every decision meets 1 ms.
The deadline-enforced tournament recorded zero automatic timeouts, errors, or
internal fallbacks. Across all 7,800 held-out candidate tables, 1,549,490 decisions
had four recorded overruns; those four occurred in the parallel strategy runs
where timing fallback was deliberately disabled. No samples are discarded.

## Validation and limitations

Unit test output: benchmarks/final_unit_tests.txt. Offline ZIP validation: benchmarks/package_check.json.
Gameplay uses only the Python standard library and packaged assets; it makes no online calls.
The simulator uses a documented local contract and forward-moving button. No official interface, payouts, or later blind schedule was supplied.
Earlier model and policy tuning used different seeds and is recorded separately in training/ and benchmarks/tuning_v1.json / tuning_v3.json.
This is a tested empirical policy, not an equilibrium solver or a claim of optimal contest performance.

The compact runtime ZIP is 707,687 bytes compressed and 2,410,698 bytes expanded,
well below the decimal 100 MB limit. Its isolated `python -I -S` probe completed
524 legal decisions with zero errors/fallbacks and zero network attempts after
the network guard self-test. All three assets and matching compiled inference
were present. This is a Python-level network-denial test, not a kernel sandbox.

All 48 final unit tests passed. The retrieved ZIP hash and every packaged runtime
file match the frozen, evaluated source. Both helpers finished; all five owned
remote jobs are terminal (four completed, one test-only failure followed by a
successful retry). No pre-existing job or shared SSH master was stopped.

Confidence intervals are descriptive and not adjusted for multiple comparisons.
The 400-table selfplay sample landed above equal share; it is a finite-sample
sanity check, not evidence of an advantage over the same policy. The final suite
contains 7,800 candidate tables and 600 independent TAG control tables. The
runtime ZIP retains the original machine-generated summary; this workspace
report adds the detailed timing interpretation above without changing the bot.
