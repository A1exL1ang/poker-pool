The simulator is an independently authored local testing environment. No
official starter kit, Agent API, later blind schedule, or payout vector was
available. Passing these tests establishes the stated local semantics only.

`engine.Hand` exposes a decision-only observation through `observation()` and
accepts `act(action)`. `play_hand` drives policies. `play_tournament` runs up to
200 hands and awards any remaining points with exact ICM. Both can collect
decision wall times and count errors, illegal actions, overruns, and fallback
checks/folds. By default strength experiments count overruns but do not enforce
them; `--enforce-time` enforces the fallback. Observations and their histories
are copied so policy mutation cannot alter engine state. Opponent hole cards,
burns, and future board cards are never exposed to a decision.

Local mechanics and conventions:

- Cards are 0..51, rank `card // 4 + 2`, suit `card % 4`.
- Stacks are integer chips behind. Pot includes all committed chips, including
  live street wagers. A call pays the smaller of the nominal deficit and the
  actor's stack. Raises specify final street totals, never increments.
- The minimum raise increment is the last full bet/raise, initially one big
  blind. Short all-ins retain that increment. Prior actors may raise again only
  when the total additional amount they face reaches a full increment; several
  short all-ins can cumulatively reopen betting. Players who have not acted
  retain raise rights. These conventions follow the public
  [Poker TDA rules](https://www.pokertda.com/view-poker-tda-rules/).
- An opening all-in below one big blind does not permit a later non-all-in
  completion: a raise must add at least one full big blind. A short preflop big
  blind does not lower the normal bring-in while two players can still bet.
  When only one player has chips, that player only owes the amount actually
  contestable against all-ins. There is no side betting when nobody can call.
  This follows the no-limit distinction in
  [Robert's Rules of Poker, section 14](https://www.pagat.com/docs/RobsPkrRules11.pdf).
- Uncalled excess is refunded at the end of each betting round. Every side-pot
  layer has independent eligibility; folded contributions remain dead money.
  Ties split integer chips and odd chips go clockwise starting left of button.
- Heads-up, button posts small blind, acts first preflop and last postflop.
  Dealing starts left of button; one card is burned before each board street.
  The transition to heads-up adjusts the button to avoid consecutive big blinds.
  The live/moving button otherwise skips eliminated seats. A dead-button tournament
  convention was not specified and is not implemented. There are no antes,
  straddles, rake, rebuys, or house interventions.
- Simultaneous eliminations are ordered by hand-start chips. Equal starting
  stacks split their tied-place payouts. This is an explicit local convention;
  the official contest's tie handling was not provided.
- A fresh policy object/reset is used for every table. Anonymous seat indices
  stay fixed within a table. Histories contain voluntary actions from the
  current hand only; blind payments are represented by commitments. History
  `amount` is raise-to for raises and actual incremental chips paid for calls.

`scenarios.py` labels every assumed payout and later blind level. Initial
2000-chip stacks, 10/20 blinds, 25-hand level changes, and the 200-hand cap come
from `RULES.txt`. Slow/medium/fast progressions and winner/top3/flat3/rank payouts
are sensitivity scenarios, not official information. Scores are normalized so
the eight-seat payouts sum to one.

Run mechanics and exact ICM tests:

```text
python -m unittest tests.test_simulator tests.test_icm -v
```

Run a bounded smoke experiment:

```text
python -m simulator.benchmark --candidate agent:Agent --reference model_equity --tables 8 --max-hands 30 --field mixed --blinds medium --payout top3 --split train --output measurements/train_smoke.json
```

Run a larger experiment only on this bot's own bounded compute allocation:

```text
python -m simulator.benchmark --candidate agent:Agent --reference model_equity --tables 1024 --field robust --blinds medium --payout top3 --split heldout --workers 8 --output measurements/heldout_robust.json
```

The `robust` field includes two copies of the local Agent plus independently
authored opponents and a matched-model equity comparator. `--opponent-config`
can fix a separate JSON configuration for imported local opponents.
`--candidate-config` and `--reference-config` accept JSON or `@local-file.json`.
`--opponents` can specify exactly seven comma-separated baseline names or local
`module:Class` imports. Baselines are checkfold, station, nit, tag, maniac,
random, equity (separate coarse heuristic), and model_equity (same trained
estimator, simple pot odds/value decisions). They are transparent stress tests,
not claims about strong equilibrium or the real contest field.

Default training seeds start at 10000, reserved held-out seeds at 90000000.
Freeze the candidate before inspecting held-out results; changing it afterward
makes that data development evidence. The CLI label alone cannot enforce this
scientific separation. Optional `--seed-start` records an explicit alternative.
Every table uses a different deck seed by default, with hero seats cycled over
eight positions. `--seating rotations` reuses each deck seed over all eight hero
seats and requires a multiple of eight tables. Confidence intervals then use
independent deck-seed block means, not correlated individual rotations. Paired
control runs use the same deck seeds/seating. Intervals are descriptive normal
95% intervals; small samples are weak evidence and an unknown opponent field
remains a major limitation. The full per-table results and options are saved
for reproducibility.
# Measurement provenance

The benchmark records SHA256 source and asset hashes at launch. It rejects
missing trained assets by default; `--allow-fallback-model` is for explicitly
labeled development smoke checks only. Seeds are separated into `train`
(10000), `validation` (5000000), and `heldout` (90000000). Select parameters
using the first two splits and freeze the candidate before using the third.

