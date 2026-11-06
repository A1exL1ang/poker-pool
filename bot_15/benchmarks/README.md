# Local simulation and evaluation

This is an independently written test environment, not the unavailable official starter kit. It implements the contract in `INTERFACE.md` and assumes standard no-limit Hold'em with no ante. Tournament starting stacks are 2,000 and the hand cap is 200 by default. The experimental blind schedule **doubles** 10/20 every 25 hands; only the initial blinds and change frequency are given in `RULES.txt`, so the doubling progression is not an official claim.

No finishing-point structure is built in. Supply a descending list through `--payouts` to run a declared hypothetical scoring experiment, or omit it to measure final chips and wins. Results with a supplied structure allocate eliminated players' points by finish and apply exact ICM to survivors at the cap. Simultaneous eliminations are ordered by starting stack; equal stacks share the affected ranks and points. That convention must be reconciled with the official simulator when available.

## Commands

```powershell
python -m unittest discover -s tests -p test_engine.py -v
python -m benchmarks.fuzz --tables 1000 --output artifacts/integration_fuzz.json
python -m benchmarks.tournament --candidate bot --tables 1000 --split tuning --population mixed --paired tight --output artifacts/tuning_mixed.json
python -m benchmarks.tournament --candidate bot --tables 1000 --split heldout --population mixed --paired tight --output artifacts/heldout_mixed.json
python -m benchmarks.tournament --candidate bot --tables 1000 --split heldout --population aggressive --payouts 50,30,20,0,0,0,0,0 --output artifacts/heldout_hypothetical_points.json
```

The `50,30,20,0,0,0,0,0` list above is solely an illustrative sensitivity scenario. Default tuning seeds begin at 10001 and held-out seeds at 900001. Avoid selecting policy parameters using the held-out results. Seeds, configuration, rows, means, standard errors, and approximate normal 95% intervals are saved. Small-sample intervals can extend outside physical bounds; they are uncertainty summaries, not probability guarantees.

The integration fuzz command rotates the candidate through all seats against varied synthetic opponents, using absent, top-three, winner-take-all and flat payout fixtures. It validates every observation, checks chip conservation and action amounts, and counts candidate internal fallbacks. It records candidate wall-clock calls above 1ms without enforcing the time limit. Those counts include host scheduling and concurrent work and should not be confused with the separate final latency benchmark. Fuzz seeds are reserved for integration checks, not held-out strategic claims; source hashes and model-presence metadata identify the checkpoint tested.

`--paired tight` reruns the same table seeds, hero seats and opponent seeds with the named baseline replacing the candidate. Each hand's shuffled deck is seeded independently of previous actions. This provides common random numbers, but different eliminations and player counts cause subsequent private cards and RNG consumption to diverge; paired runs are not identical trajectories.

`--enforce-1ms` applies a local wall-clock limit to every agent call, replacing an overrun with check/fold. It cannot reproduce unknown contest hardware or scheduler accounting. Aggregate errors, invalid actions and overruns include all players. Dedicated candidate-only latency measurements should be reported separately. Observation construction and observer callbacks are outside the call timer.

## Python APIs

`play_table(agents, seed, ..., blind_schedule=[(sb,bb),...], payouts=None, observer=None)` permits custom populations, schedules and monitoring. Every agent is reset at the beginning of every table. `observer(observation, action, elapsed_ns)` sees public state, the acting player's own cards, the returned action and decision-call timing; it never receives another player's private cards or the deck.

`evaluate_candidate(factory, tables, seed, population, max_hands, payouts, ...)` in `benchmarks/tournament.py` runs the candidate through all eight seats against deterministic synthetic populations: mixed, tight, loose and aggressive. These opponents are independent heuristic stress tests, not models of actual contest entrants. Positive performance against them cannot establish equilibrium quality or optimality.

`benchmarks/challenge.py` adds separate equity-aware held-out-policy challenges: `EquityBaseline(style='tight'|'balanced'|'loose', seed=...)` and `PolarizedBettor(seed=..., bluff_frequency=.22, bet_fraction=1.25)`. Their policies use position-based opening widths, equity-table pot odds, side-pot weighting, fixed adjustments for facing aggression, and value/bluff betting. `PolarizedBettor` chooses large value bets and weak-hand bluffs; its bluff frequency is a probability conditional on an eligible weak hand, not the proportion of all bets that are bluffs. Parameters were selected independently, without playing or tuning against the candidate. They have no ICM adjustment or opponent adaptation, and are not GTO. Their use of the same equity assets introduces correlated model errors, so they provide additional strategic stress rather than fully independent truth.

For a held-out challenge, pass an `opponent_factory(i, table_seed)` returning these agents to `evaluate_candidate`; keep the candidate frozen when inspecting the results. `python -m unittest discover -s tests -p test_challenge.py -v` checks their interface, conservation, legal play against themselves and older baselines, reset determinism, guaranteed chops and side-pot equity weighting. Those tests do not run the candidate.

## Mechanics and verification

The simulator handles heads-up blind/action order, dead-button progression, dead small blinds, nominal big-blind bring-in when the BB is short, per-player reopening after cumulative short all-ins, side pots, uncalled excess, folded-hand exclusion, ties and clockwise odd-chip assignment. Call amounts are capped by stack; raises specify total commitment on the street. Illegal actions and exceptions receive the local check/fold fallback.

Tournament mechanics were checked against [Poker TDA 2026 rules](https://www.pokertda.com/view-poker-tda-rules/), especially rules 21, 23, 34, 36, 45 and 49. A prior check does not restore a right to raise against an opening all-in below a full minimum bet; an unacted player must add a full minimum raise. The TDA's [short opening all-in discussion](https://www.pokertda.com/forum/index.php?topic=663.0) clarifies that edge case. These are public references for the local rules interpretation, not proof of official contest-interface compatibility.

`tests/test_engine.py` includes an independently implemented five-card reference and exhaustive five-card-subset comparisons for 1,500 randomly generated five-to-seven-card hands, plus 5,000 differential comparisons against the bot's runtime evaluator. It also checks rule-critical action sequences, conservation, ICM examples, simultaneous elimination, reset-compatible deterministic runs, and error/time-limit fallbacks. The independent differential check found and led to fixing a three-pair kicker bug in the runtime/native evaluators before postflop model training.
