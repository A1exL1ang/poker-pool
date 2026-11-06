# bot_20 — independent offline NLHE bot

This workspace contains one Python poker bot, its own simulator, generated
equity assets, tests, and reproducible experiments. **It is local development,
not a contest submission or a claim of official interface compatibility.** No
official starter kit, Agent signature, payouts, or later blind levels were
available. The orchestrator must adapt the explicit local contract to the
actual interface before evaluation/submission.

## Run

Gameplay needs only Python's standard library. Python 3.11 on Windows and 3.12
on Linux are tested. Construct the agent once before timed play:

```python
from agent import Agent

bot = Agent(seed=20)
assert bot.estimator.preflop_loaded and bot.estimator.model_loaded
action = bot.act(observation)
# At every new table, use a fresh Agent or call bot.reset(new_table_id).
```

`pokerbot/interface.py` defines the complete local `Observation` and `Action`
types. Do not infer an official mapping from these names. `validate_observation`
and `is_legal` provide integration checks outside the hot path.

- Cards are integers 0..51: rank `card // 4 + 2`, suit `card % 4`.
- Streets 0,1,2,3 mean preflop, flop, turn, river.
- `stacks` are chips behind. `committed` is each player's total investment this
  hand; `street_committed` is investment on this betting round.
- `pot` includes all committed wagers, including the wager being faced.
- `to_call` is the **uncapped** nominal deficit. A call pays `min(stack,to_call)`.
- A raise `amount` is its final street total, **raise-to**, not additional chips.
  Call amounts are the actual incremental chips; check/fold amounts are zero.
- The host supplies `can_raise`, `min_raise_to`, `max_raise_to`. A short raise
  below minimum is allowed only when it is the player's full all-in.
- `history` is public action history for the current hand. Opponent private
  cards, future board cards, and opponent team identities must never be passed.
- `payouts`, if supplied, must be the actual nonincreasing finishing-position
  awards for all starting seats. With `None`, the policy uses chip utility.
- Use a fresh agent, `reset()`, or a changing `table_id` for each table. No
  behavioral memory is written to disk. Diagnostic counters are lifetime-only.

All expensive imports, lookup construction, asset parsing, and numeric tree
compilation occur before `act()`. **If the real host includes startup inside the
1 ms budget, the initialization arrangement needs to change.** The unknown
official interface prevents an unconditional timing-compatibility promise.

## Strategy

The bot combines a generated 169-class preflop equity table, position and stack
dependent opening ranges, short-stack shove estimates, public-action range
models, selective value betting and bluffs, and pot-odds calls. Anonymous-seat
statistics use strong priors and count only observed opportunities. They reset
at each table; there is no team identification or coordination.

Postflop equity comes from compact trees trained on independently generated
native Monte Carlo deals. Labels include blockers, ties, multiple opponents,
preflop range tightness, and public-board value/draw selection. Inference is
bounded pure Python; there are no live rollouts or network calls. See
`training/README.md` for the exact range model and sampling limitations.

The policy caps the pot it can win when calling short. With supplied payouts,
single-opponent all-in, river, and nearly committing calls use exact ICM values
of constructed fold/win/loss stack outcomes. The outcome model is binary and
does not solve future betting or general tie distributions. Small calls and
multiway decisions use a documented heuristic risk premium. Guaranteed shared
unbeatable boards bypass the binary model. Large value raises also undergo an
ICM check, so a value branch cannot simply ignore an extreme bubble.

This is an engineered and empirically tested policy, **not a solved equilibrium
strategy or a claim of absolute optimality**. Equity accuracy against our range
family does not establish that real opponents follow that family.

## Tests and experiments

```sh
python -m unittest discover -s tests -v
python training/validate_evaluator.py
python -m benchmarks.latency --output measurements/latency_reproduction.json
python -m simulator.benchmark --candidate agent:Agent --reference model_equity --tables 256 --field robust --blinds medium --payout top3 --split heldout --enforce-time --output measurements/reproduction.json
```

The benchmark's `heldout` option is for reproducing a frozen candidate. Do not
use it for further tuning and then describe those results as held-out. Splits,
selection rules, and timing conventions are recorded in
`benchmarks/protocol.json`. Reports distinguish development, validation, and
final evidence. See `reports/RESULTS.md` for completed final measurements.

The simulator covers no-limit raise reopening, cumulative short all-ins,
partial blinds, side pots, refunds, ties/odd chips, heads-up transitions, and
ICM cutoff scoring. Tests include an independent hand-ranking reference,
exhaustive five-card category counts, independent ICM finish-order enumeration,
adversarial accounting cases, and randomly generated legal action sequences.
Its own limitations and assumptions are in `simulator/README.md`.

The contest rules specify 2,000 starting chips, eight seats, 10/20 opening
blinds, 25-hand blind levels, and a 200-hand cap. **Every later blind level and
every payout vector in our local scenarios is hypothetical.** Benchmarks vary
these assumptions; they do not invent official scoring. Local opponents are
independently authored transparent baselines and copies of this bot, not other
teams' bots or private repositories.

## Rebuild the equity assets

Training requires a C++17 compiler, NumPy, and scikit-learn; these are not
gameplay dependencies. `training/run_training.sh` reproduces the initial model.
Subsequent selected-model commands are recorded with the final results and
model metadata. Training inputs and model selection use different seeds from
final equity and tournament evaluations. Independently sampled deals can still
share equivalent card states; no claim of perfect state-disjointness is made.

All generated runtime assets are under `pokerbot/assets`. Missing or malformed
assets expose `model_loaded=False` or `preflop_loaded=False` and retain a crude
development fallback. The final package verifier rejects that condition.

## Rules, provenance, and scope

The complete supplied `RULES.txt` was read. The user's local instructions
explicitly replace the event's 60-minute build window with no fixed local
deadline. Gameplay remains offline, the runtime package remains below 100 MB,
and errors/timeouts use check-or-fold in the testing host. The actual timing
tails and fallbacks are retained in measurement files.

No other bot workspace, parent project, earlier team experiment, private poker
repository, or other team's artifacts were accessed. At most two helpers
worked on this same bot. Remote jobs use only the authorized private directory
and new bounded `bot_20` jobs; job IDs and completion records are in `.control`.
No pre-existing allocation was inspected, attached to, modified, or stopped.
Nothing was pushed, published, or submitted.

Public sources consulted for concepts and mechanics, without copying bot code:

- Brown and Sandholm, [Superhuman AI for multiplayer poker](https://noambrown.github.io/papers/19-Science-Superhuman.pdf): offline abstraction/self-play context; our implementation does not reproduce Pluribus.
- [Poker Tournament Directors Association rules](https://www.pokertda.com/view-poker-tda-rules/): standard betting, reopening, side-pot and tie mechanics; these are not a substitute for the unavailable contest engine.

Unresolved integration questions are the official observation/action API,
runtime version and initialization accounting, actual payout and blind
schedule, and engine-specific conventions such as button movement and tied
eliminations. They are explicit adapter/test tasks, not hidden assumptions of
compatibility.
