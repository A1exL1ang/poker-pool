# Measurement protocol and evidence

This workspace is a local experiment, not an official contest submission.
Payouts, blind increments, observation API and sparring policies are local
assumptions. Every reported tournament uses eight bots, 2,000 starting chips,
initial 10/20 blinds, changes every 25 hands and a 200 hand cap. Point vectors
are explicitly synthetic and normalized to total 1. An equal share is 0.125.

## Data and model separation

`training/data_manifest.json` records source/data hashes and generator seeds.
There are 300,000 training situations with 1,024 simulations per range target.
Complexity selection fits train0..4 and evaluates training holdback train5.
The selected final model is fitted to all six training files. The independent
15,000-state set uses a different seed and 4,096 simulations per target and is
never fitted. The original full forest and final joint model each evaluated
that same heldout set once; those results did not select the joint settings.

The final 140-tree joint model's learned equity MAE is 1.457%/1.467% across the
whole heldout set. Flop MAE is 1.090%/1.144%; turn MAE is 1.396%/1.367% for the
uniform and top 20% ranges respectively. Production river decisions bypass
the forest. Exact river probabilities and ties were compared with all 990
physical opponent combinations on 1,000 newly seeded states and matched
exactly; all suit textures and rare shared-nuts/flush/full-house cases have
additional tests. The fixed ranges are not posteriors conditioned on betting.

All 2,598,960 five-card combinations match the known category counts. Random
five/six/seven-card ranks are compared with an independent best-five subset
enumerator. These are not exhaustive seven-card or full-policy proofs.

## Policy selection

`tuning_round1.json` is exploratory selection evidence: 12 predeclared parameter
variants, seven synthetic scenarios, 128 tables per scenario, each with a paired
original-parameter reference. It predates the final joint/exact-river engine
and the observed-action denominator fix. Its initial seat rotation correlated
with the dealer, starting hero on the button. Do not treat it as final evidence.

`validation.json` contains four candidates on seven fresh-seed scenarios with
512 tables each, using the final equity engine. The balanced candidate led
the equally weighted average by 0.00819 points/table over original parameters
(SE 0.00274). This is selection data, with multiple comparisons and the same
initial-position limitation. It is not a heldout result. Two other candidates
each produced one internal fallback among 1,774,618 candidate decisions: an
over-one range-mixture weight could produce negative equity. The final policy
clamps this weight and has targeted regressions.

Final defaults were selected before accessing the final test seeds. The final
validation rerun and heldout harness balance every physical seat and initial
dealer-relative position once per 64 table block. Rival styles are shuffled.
Deck seeds are shared in paired runs, although action-dependent eliminations
can change later card assignment. Exact runtime source hashes are recorded.

## Final remote evaluation

The final remote run passed all 65 regression tests. Mechanics fuzzing covered
10,000 hands plus 100 tournaments and 80,471 decisions, with zero invariant,
illegal-action or exception failures. The corrected validation rerun covered
1,756,923 candidate decisions with zero illegal actions, exceptions or internal
fallbacks. Runtime source hashes were stable.

The frozen policy's primary heldout evaluation covered 16,384 paired table
cases on untouched seeds beginning at 1,000,000. Across the 16 equally weighted
scenarios, its mean normalized points were 0.23812, gaining 0.005844 over the
original parameters (SE 0.001586; approximate 95% CI 0.002736 to 0.008951).
The four TAG audits added 4,096 paired cases. Across all 20,480 candidate tables
and 2,570,306 decisions, there were zero illegal actions, exceptions or internal
fallbacks. No policy or model changes were made from these test results.

Results do include losses. Against calling stations the delta was -0.01992
(SE 0.00677); against frozen original-parameter self-play it was -0.00410
(SE 0.00646), and with fast blinds it was -0.00059 (SE 0.00559). The self-play
result does not establish an advantage over the frozen reference. All four
separately seeded TAG audits were positive, including +0.07738 in the mixed
field and +0.05400 in self-play. Full results and per-scenario confidence
intervals, including the losses, are in `heldout.json`.

The 16 predeclared scenarios comprise mixed, tight/aggressive, nit, loose,
maniac, calling-station, shove, random, frozen self-play and independently
perturbed ABC fields; gentle/fast blinds; and winner-only, equal top three,
top four and linear point vectors. Each uses 1,024 paired tables. Four additional
audits compare against a simple TAG hero on separate seeds beginning 2,000,000.
All scenario results, including losses, are retained. The equally weighted
scenario average is an arbitrary development summary, not a contest-score
forecast. Scripted policies share the equity utility, limiting independence
and realism of the opponent models. Frozen self-play uses original parameters.

## Timing

The earlier serial Windows frozen-engine/original-parameter replay recorded
100,000 decisions: p99 279.8 µs, 15 wall-clock overruns above 1 ms. There were no
illegal actions or internal fallbacks. Fresh-Agent/cache-cold replay had 0/2,000
overruns; artificial eight-seat uncached-ICM stress had 4/2,000. These are prior
measurements, not the final selected-policy timing.

The final serial protocol records 100,000 replay decisions, 2,000 cold-state
decisions and 2,000 separately labeled artificial ICM stress decisions, with
per-street percentiles, maxima and every overrun. A 512-table paired trial
compares the same policy with/without automatic 1 ms check/fold enforcement;
enforcement applies to all seats. This separates policy evaluation from
deadline-driven action changes. Parallel tournament timing is not used as
deadline evidence. Source hashes, interpreter, OS and load notes accompany it.

On Linux Python 3.12.7, the final serial 100,000-decision replay measured
p50 22.7 µs, p99 115.8 µs, p99.9 166.0 µs and maximum 870.9 µs, with zero
overruns above 1 ms. The 2,000 cold-state decisions also had zero overruns
(p99 139.7 µs, maximum 763.7 µs). The 2,000 uncached eight-seat ICM stress
decisions had zero overruns (p99 264.1 µs, maximum 364.4 µs). All three had
zero illegal actions and internal fallbacks.

The 512-table same-policy deadline trial measured 55,118 hero decisions in
each arm, with zero timeouts at any seat, zero illegal actions, exceptions or
internal fallbacks, and identical points with/without deadline enforcement.
These are observations, not a hard deadline guarantee. The eight-worker
heldout runs separately recorded 183 wall-clock overruns among 2,570,306 hero
decisions, maximum 22.23 ms; deadlines were not enforced for that policy-score
experiment. The earlier Windows overruns above are also retained. Scheduling
and hardware therefore matter despite the substantial serial timing margin.

Module import/model loading, Agent construction, observation construction and
JSON transport are outside the act() timer. OS scheduling pauses remain inside
wall time. Python and a shared OS cannot guarantee a hard 1 ms deadline. The
official sandbox, hardware and adapter are unavailable. CPU per-act clock
resolution on Windows is coarse, so wall timings are the primary measurements.

## Reproduce

Run substantial commands only on a new bounded authorized compute allocation;
the user has requested a quiet laptop. There is no heavy gameplay computation.

```text
python -m unittest discover -s tests -v
python benchmarks/fuzz_mechanics.py --hands 10000 --tournaments 100 --seed 31291 --output measurements/mechanics_final.json
python benchmarks/validate.py --tables 512 --workers 8 --output measurements/validation_final.json
python benchmarks/heldout.py --tables 1024 --workers 8 --params measurements/selected_policy.json --tag-audit --output measurements/heldout.json
python benchmarks/latency.py --refresh-corpus --corpus-tables 64 --warm-decisions 100000 --cold-decisions 2000 --stress-decisions 2000 --enforce-deadline --deadline-tables 512 --params measurements/selected_policy.json --output measurements/latency_linux_final.json
python package_bot.py
```

The corrected current source will not reproduce the earlier exploratory run
bit-for-bit. Historical reports identify their runtime hashes and parameters.
Final measurements use the frozen selected defaults and corrected harness.
Normal-approximation confidence intervals use 1.96 × sample SE and are not adjusted
for multiple tests. No absolute optimality or official compatibility is claimed.
