# Local Hold'em environment

This is an independently written test environment. No official starter kit,
observation format, point vector, or complete blind schedule was available.
It does **not** establish official interface compatibility or contest strength.

`RULES.txt` provides the contest-level facts: eight players, 2,000 chips each,
10/20 initial blinds increasing every 25 hands, 200-hand cap, and ICM allocation
of remaining finishing points. The complete blind levels and finishing points
below are explicitly synthetic and configurable.

## Observation and action contract

The bot implements `act(observation_dict) -> action_dict`. Optional `reset()` is
called once at the beginning of a local table. The benchmark constructs fresh
agents using `Agent(seed=integer, **kwargs)`.

Cards are integers `0..51`: `rank = card // 4 + 2`, `suit = card % 4`. Suit names
are immaterial; tests use clubs, diamonds, hearts, spades in that order. `hero`
and `button` are seat indices. `players` includes every original seat in index
order. Eliminated seats have zero chips and `folded=True`. Seats are anonymized
within a table and expose no cross-table identity.

`players[i].stack` means uncommitted chips. `bet` is the player's current-street
contribution, already included in `invested` and `pot`. `invested` is the entire
hand contribution. `pot` includes all current bets. `to_call` is the additional
payment capped by hero's remaining stack. The full current wager can be derived
from the maximum current bet, with the nominal full big blind applying preflop
even when a short blind posts less. Against only all-in opponents, the actor
matches their actual wager and cannot raise.

The action strings are `fold`, `check`, `call`, and `raise`. For `raise`, `amount`
is the **total street contribution**, not an increment or a hand-total amount.
`min_raise_to` is the full minimum raise-to and can exceed `max_raise_to` when
only a short all-in is possible. Such a short raise must equal `max_raise_to`.
`can_raise` incorporates player-specific reopening and the presence of an
opponent with chips who could respond. A no-bet call is normalized to a check.
Other illegal actions produce a counted check/fold fallback during tournament
play; direct `Hand.apply()` instead raises `IllegalAction` without changing state.

History entries contain `seat`, `street`, `kind`, `amount`, `raise_to`. `amount`
is the chips paid by that action; `raise_to` is that seat's resulting street
contribution even for calls/blinds/checks. Blind kinds are `small_blind` and
`big_blind`. The hero sees only its own hole cards and the exposed board. Returned
observations are detached copies: modifying them cannot change engine state.

`payouts` is `None` when unavailable; otherwise it is an explicitly supplied,
nonincreasing vector of points for first through last place at the original
table. `hands_remaining` includes the current hand. Frozen documentation
dataclasses and `as_dict()` are in `schema.py`.

## Local mechanics and assumptions

The betting implementation follows the minimum-raise, cumulative short-all-in,
heads-up action-order and odd-chip conventions in the
[Poker TDA 2026 rules](https://www.pokertda.com/view-poker-tda-rules/), checked
November 6, 2026. This is a selected local standard, not a representation of an
unseen contest implementation. Per-player last-action wager levels determine
whether cumulative short raises reopen action; a check does not reopen against
a subminimum opening all-in unless a full wager is subsequently faced. The
[TDA administrator's historical clarification](https://www.pokertda.com/forum/index.php?topic=746.0)
also supports the short-opening case. Physical-deck dealing includes burn cards.

No-limit bets, all-in calls, side pots, folded money, exact ties, and uncalled
refunds are implemented. Odd chips go to tied winners in clockwise order left
of the button. Tournament blinds use a dead-button/dead-small-blind progression
after eliminations; heads-up transitions advance the big blind to avoid giving
the same surviving player consecutive big blinds. Simultaneous eliminations
rank larger starting stacks higher; equal starting stacks share affected point
awards. There are no antes, rake, straddles, rebuys, or late entries.

At the hand cap, remaining top-place awards are distributed by exact subset-DP
ICM. ICM models finish order from chip proportions, not skill. The default local
blind sequence is `10/20, 15/30, 25/50, 50/100, 75/150, 100/200, 150/300,
200/400`; the final level repeats if more than 200 hands are requested. Synthetic
point vectors include top-three `[50,30,20,0,0,0,0,0]`, winner-only
`[100,0,0,0,0,0,0,0]`, graded `[35,25,18,12,6,3,1,0]`, and equal top-four
`[25,25,25,25,0,0,0,0]`. These are sensitivity scenarios, **not official payouts**.

## Reproducible validation and benchmarks

Run from the workspace root:

```text
python -m unittest discover -s tests -p "test_sim*.py" -v
python -m sim.benchmark --candidate baseline:tag --tables 100 --seed 73400 --out sim/validation_measurement.json
python -m sim.benchmark --candidate agent:Agent --tables 200 --seed 10000 --suite mixed --payout top3 --out measurements/tune_mixed.json
python -m sim.benchmark --candidate agent:Agent --tables 200 --seed 80000 --suite aggressive --payout winner --enforce-timeout --out measurements/heldout_aggressive.json
```

Reserve evaluation seeds before tuning, and do not describe exploratory seeds
as held out. `--kwargs` accepts JSON constructor parameters; `--compare-kwargs`
runs a second variant with matching seeds, hero seat rotation, and baseline
seeds, reporting paired point differences. Initial decks are shared but future
game paths may diverge. `--capture path.jsonl` writes up to `--capture-limit`
hero observations/actions for diagnostics. `--hide-payouts` tests missing-point
structure behavior. All output contains the assumptions and per-table rows.

The independent baseline styles are calling station, nit, tight aggressive,
loose aggressive, maniac, randomized, and push/fold. Each uses a deliberately
coarse hand-strength heuristic and does not call the candidate policy or its
trained equity estimator. Suites are `mixed`, `calling`, `tight`, `aggressive`,
and `balanced`. A separate `challenge` suite was defined after exploratory
opponent tuning and reserved for final held-out evaluation. It consists of
three range-aware opponents, two trapping opponents, and two pressure opponents.
These independently implemented policies do not import the candidate policy;
they do share the learned equity estimator, which limits their independence
from modelling errors. Their stronger hand selection, postflop likelihood
updates, trapping, and polarized pressure are intended to challenge weaknesses
missed by the simple baselines. They are not equilibrium strategies. Only
challenge-versus-challenge legality checks were run before reserving the suite;
candidate-versus-challenge performance was intentionally left unmeasured.
Beating these opponents is useful local evidence, not proof of
equilibrium play or expected contest ranking.

The `selfplay` suite is available for **tuning**, using seven clones of this
same bot with explicit fixed parameter dictionaries: initial, wider opening
(`1.2`), stronger action selection (`1.8`), more aggression (`1.6`), cautious,
no opponent adaptation, and loose. These are this team's own policy variants;
no other bot source or artifacts are used. The exact dictionaries are included
in every selfplay measurement. Source and model SHA-256 hashes are recorded,
and a file-change flag identifies edits occurring while a benchmark runs.

Timing encloses only `agent.act()` and includes scheduler pauses; construction,
observation creation, action validation, and callbacks are excluded. Measured
overruns are always counted. With `--enforce-timeout`, the returned action is
replaced with check/fold above 1 ms. The simulator cannot preempt a bot that
never returns. Candidate errors, illegal actions, and timing failures are
reported separately. `candidate_fallbacks` additionally reports the candidate's
internal `Agent.errors` counter, since caught internal errors never reach the
simulator's external-exception count. Confidence intervals are approximate normal intervals
using independent tables as the sampling units.

`validation_measurement.json` is a 100-table simulator smoke/robustness run,
**not a candidate performance result**. It uses `baseline:tag` and seeds
73400–73499. The measurement file records machine-dependent timing exactly.
