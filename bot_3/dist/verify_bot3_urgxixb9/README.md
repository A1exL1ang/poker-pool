# bot_3 — independent no-limit Hold'em bot

One Python bot, authored in this workspace. Gameplay uses only the Python standard
library and local generated assets. No network calls, model API calls, subprocesses,
or Monte Carlo rollouts occur inside `Agent.act`.

**This is a working local implementation, not an officially compatible submission.**
`RULES.txt` was read in full. Its official Agent interface, payout vector, and blind
schedule increments were not supplied. An adapter to the real event interface is
still required. Nothing has been submitted, committed, pushed, or published.

## Run and test

From this workspace, with Python 3.10+ (tested with Python 3.11.9):

```powershell
Get-Content -LiteralPath examples/observation.json | python -m bot3
python -m unittest discover -s tests -v
python -m tools.benchmark --latency 100000 --output evidence/latency.json
python -m tools.benchmark --tables 400 --suite mixed --seed 620671 --output evidence/mixed.json
python -m tools.package_bot
```

The example returns `{"action":"raise","amount":48}`. JSON input is one complete
observation per line; one decision is written per line. Keep the process alive for
the table. `{"command":"reset","table_id":"new-id"}` clears table state.
The JSON adapter validates input/output; use the direct method for tight timing:

```python
from bot3 import Agent
agent = Agent(seed=3)       # load once, outside the action timer
decision = agent.act(observation)
agent.reset("next-table")  # also automatic when table_id changes
```

There are **no pip dependencies**. The attempted numerical-package download was
blocked, so all offline numerical tooling was implemented in local C++ instead.
No Modal or Engaging jobs were launched; local compute was sufficient. The inbox
reported that Engaging authentication was unverified. Existing jobs/allocations
were not inspected or touched.

## Strategy

- Preflop: generated equities for all 169 hand classes against six range widths,
  positional opening ranges, value reraises, selected ace-blocker reraises,
  realization discounts, effective-stack sizing, and short-stack commitment.
- Heads-up at at most 12 effective big blinds: an offline CFR+ solution to the
  **restricted** SB shove/fold and BB call/fold game, with 0.5/1 blinds and no ante.
  This is not a Nash solution to full no-limit poker. Other actions and deeper
  stacks use the ordinary policy.
- Postflop: an 80-tree, depth-eight equity model trained on 90,000 generated
  observations, each with 1,024 rollouts. Six outputs model random, three preflop
  ranges, a continuing range, and a stronger value range. Exact current best-five
  strength, board texture, draws, blockers, position, pot odds, stack-to-pot ratio,
  action history, and table-local seat statistics determine betting/calling.
- Multiway equity uses a power approximation; eligible-pot caps avoid crediting
  side pots that hero cannot win. Shared-board and royal-flush cases have guards.
- If actual payouts are supplied, material heads-up calls with at most five
  surviving players use an exact ICM calculation on simplified fold/win/lose
  outcomes. No payout vector is assumed by default; default decisions target chip
  EV. ICM does not model an entire future tournament or arbitrary multiway pots.

Random mixing is reproducible with the seed. Seat statistics reset per table and
do not identify teams or use information from other tables.

## Local interface contract (v1)

`examples/observation.json` is a complete valid example. Clockwise seats are in
increasing integer order, wrapping at the end. Supply every table seat, including
folded and eliminated seats. Required fields:

- `hero_seat`, `button`, `hand_id`, `street`, `hole_cards`, `board`, `players`.
  Street is `preflop`, `flop`, `turn`, or `river`; board length is 0/3/4/5.
  Cards are strings such as `As`, `Td`, or integer IDs `rank*4+suit`, with ranks
  `23456789TJQKA` and suits `cdhs`. Hero has exactly two cards; no duplicates.
- Each player: `seat`, `stack` (chips behind), `bet` (this street's contribution),
  `committed` (all contributions this hand, including `bet`), `folded`, `all_in`.
  Mark an eliminated seat folded and not all-in. All chip amounts are integers.
- `pot` includes **all** contributions already made, including current wagers.
  `big_blind` is the current full big blind. `current_bet` is the full street bet
  to match; supply it explicitly, especially with a short all-in big blind. If
  absent, the bot assumes the maximum posted player bet. `to_call`, if supplied,
  is the additional call cost capped at hero's chips behind.
- `legal_actions` contains strings from `fold`, `check`, `call`, `raise`; `raise`
  also represents an opening postflop bet. The engine must remove `raise` when
  action has not reopened. `min_raise_to` is the total street commitment of a
  minimum full raise; `max_raise_to` is the largest allowed total. A below-minimum
  raise is used only as a whole-stack all-in. These bounds are required even if a
  raise is unavailable.

Optional: `table_id` (recommended), `history` (append-only full **current-hand**
events, chronological), `payouts` (descending actual finishing-place point values),
and `ante` (defaults to zero; nonzero disables the no-ante heads-up tables).
History events contain `seat`, `street`, `action`, `amount`; `pot_before` is
optional. Blind posts are not voluntary `raise` events. Hand IDs must change at
each new hand. Reset the agent before a new table if `table_id` is omitted.

Output is `{"action": name, "amount": integer}`. Fold/check amounts are zero;
call amounts are **additional chips**; raise amounts are **total street chips**.
`agent.py` merely re-exports the same local `Agent`; it is not an official adapter.

## Evidence and timing limits

The reproducible evidence lives in `evidence/`. Validation includes an independent
five-card enumeration oracle, native/Python feature parity, suit invariance,
generated/JSON model parity, randomized action legality, state reset, ICM,
heads-up order, cumulative underraises/reopening, all-ins, side pots and odd chips.

The chosen model's independently seeded 9,000-hand validation set uses 4,096
rollouts per observation. MAE across its six targets is **1.77%–2.46%**; the target
itself is a noisy, explicitly modeled opponent range, not ground truth about a
contest opponent. `evidence/model_validation.json` records error distributions.
The push/fold solver's best-response gap is below 0.000001 BB/hand **on its sampled
matchup matrix**; sampling error and full-game deviations are not in that number.

A 100,000-decision Windows timing run measured **110 us mean, 310 us p99,
822 us p99.9**; **60 decisions exceeded 1 ms**, maximum 13.3 ms. It excludes import,
construction, JSON and validation. The host is not real time; this is **not a
guarantee that every contest call meets the one-millisecond rule**. CPU accounting
on this Windows installation is too quantized for per-call CPU percentiles.
See the final timing JSON for subsequent measurements and exact source hashes.

In a 400-table mixed scripted-opponent fixture, the bot won **135/400 (33.75%)**;
approximate 95% interval 29.1%–38.4%. Equal-share reference is 12.5%. Earlier
balanced and pressure suites won 83/300 and 105/400 respectively, with prior model
versions. These are synthetic sanity checks, not official scores or evidence of
superiority to competition bots. Opponents are simple test policies written here.
The local simulator starts eight players at 2,000 chips and 10/20, **illustratively
doubles blinds every 25 hands**, and caps games at 200 hands. The doubling rule is
an assumption; only initial blinds and the 25-hand interval are in `RULES.txt`.

## Rebuild all generated assets

Runtime assets and generated Python are already present. To reproduce them with
g++ and OpenMP (tested with g++ 14, eight bounded local threads):

```powershell
g++ -O3 -std=c++17 -fopenmp tools/offline.cpp -o tools/offline.exe
.\tools\offline.exe all 80 8
g++ -O3 -std=c++17 -fopenmp tools/pushfold.cpp -o tools/pushfold.exe
.\tools\pushfold.exe
python -m tools.export_model
python -m unittest discover -s tests -v
```

On Linux, omit `.exe`, use `./tools/offline` / `./tools/pushfold`, and otherwise use
the same commands. Generation is seeded and parallel work has disjoint outputs.
The training/validation binary files and matchup matrix are retained locally.
`tools.package_bot` includes only the runtime sources, two small JSON tables,
README, example, and checksums in `dist/bot_3.zip`; it verifies an extracted copy
with isolated Python. It does not contact the competition.

Known limits: heuristic betting and opponent ranges, approximate multiway equity,
no full-game equilibrium search, no official-engine validation, unknown payouts
and blind increments, environment-dependent timing, and unverified strength
against real entrants. Recheck the real interface, rules, action semantics and
timer on receipt of the official kit before any submission.
