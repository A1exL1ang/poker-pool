# bot_14 — independent offline no-limit Hold'em bot

This workspace contains an original Python bot, its assets, a local poker engine,
tests, and reproducible experiments. The entry point is `agent.py`:

```python
from agent import Agent
bot = Agent(seed=140014)
action = bot.decide(observation)
```

The selected default configuration is recorded in
`measurements/selected_config.json`. It uses the fast native forest when
available, exact heads-up river enumeration, 15% wider opening ranges, and a
7 BB open-shove threshold. The constrained heads-up shove/fold option is disabled
by default after full-game development comparisons; it remains configurable.

**No official starter kit, Agent interface, payout vector, or later blind levels
were available. This is a local implementation, with no claim of official
interface compatibility. Nothing has been submitted, pushed, or published.** An
official adapter must map the documented chip and action semantics below.

## Runtime and packaging

Gameplay requires Python's standard library and the included `assets/` files.
There are no gameplay network calls, external services, subprocesses, package
installation, compilation, training, or file writes. Assets and an optional
native library are loaded during import/initialization.

The portable Python path runs without the native library. Generic x86-64 Windows
and Linux libraries can accelerate the same equity model and exact river
enumeration. An absent or incompatible library selects the Python fallback.
The native source is `tools/runtime_equity.cpp`; packaged builds do not use
`-march=native`. Python 3.11 on Windows and Python 3.12 on Linux are the tested
environments. Import/startup time is separate from the measured decision time;
the contest's unavailable loader must be checked during official integration.

The collected runtime archive is **1.40 MB** and its source/asset hashes match
the frozen candidate. All 105 tests passed. In 2,000 held-out mixed-field tables,
mean synthetic normalized points were **0.35244** (95% interval
0.34395–0.36093; equal-share reference 0.125). Native replay p99 was **168 us**,
with **2 of 96,560 calls above 1 ms**. Full evidence and timing limitations are
in [`measurements/README.md`](measurements/README.md).

Keep `agent.py`, the `pokerbot/` package and the three selected JSON assets
`preflop.json`, `pushfold.json`, and `equity_model.json` together. The generated
runtime archive and SHA-256 manifest are described in `measurements/README.md`.
Training data, remote access configuration, job metadata and checkpoints are
excluded from that archive.

## Local observation contract

Cards are integer IDs `4 * (rank - 2) + suit`, where ranks are 2 through 14 and
suits are `c,d,h,s` numbered 0 through 3. Thus `2c=0` and `As=51`. Suits have no
tie-breaking order. `pokerbot.interface.card("As")` converts readable notation
outside the timed decision call. Input cards must be valid and unique.

The observation is a mapping with these fields:

- `hero`, `button`, `small_blind_seat`, `big_blind_seat`: physical seat indices.
  The small-blind seat can be `None` under the local dead-button rule.
- `hand_id`: monotonically increasing within a table; `street` is `preflop`,
  `flop`, `turn`, or `river`.
- `hole_cards`: only the hero's two cards. `board`: zero, three, four or five
  community cards. Opponent cards and the deck are never needed or inspected.
- `players`: physical-seat order, including eliminated seats. Each mapping has
  `stack` (chips remaining behind), `committed` (current street contribution),
  `invested` (whole-hand contribution, including `committed`), `folded`, and
  `all_in`. `starting_stack` is the stack before this hand and is recommended.
- `pot`: sum of all whole-hand contributions, including the current street and
  folded players. It excludes chips remaining behind. Uncalled upper chips
  are returned by the engine; the bot limits risk to contestable contributions.
- `current_bet`: current nominal street commitment to match. Before the flop
  this is at least the nominal BB even if that blind posts short.
- `to_call`: **uncapped** `max(0, current_bet - hero.committed)`.
- `min_raise_to`: current bet plus the last full raise; `can_raise` includes
  reopening rights and the presence of an opponent able to contest a raise.
- `small_blind`, `big_blind`: integer chips for this hand.
- `history`: public events for this hand, in order. Each has `seat`, `street`,
  `action`, `amount`, and `paid`. A raise's `amount` is a street total; a call's
  amount is additional chips paid. `paid` always changes the pot, and can be
  negative for an uncalled return. Optional `to` is the resulting street total.
  Blind actions are `post_sb` and `post_bb`; other actions are `fold`, `check`,
  `call`, `raise`, and `return`.
- Optional `payouts`: first-place through last-place **points**, including
  zeros, for the original table. Without this field, the bot uses linear chip
  value. Optional `hand_start_stacks`, `initial_stacks`, and `max_hands` are
  supplied by the local engine for diagnostics and adaptation.

The engine also supplies `legal_actions`, `max_raise_to`, `last_full_raise`, and
`starting_stack` at top level. Its observations are private, serializable
snapshots. The bot does not rely on fields containing hidden opponent data.

## Local action contract

The return value is one of:

```python
{"action": "fold"}
{"action": "check"}
{"action": "call"}
{"action": "raise", "amount": 120}
```

A raise amount is the **total hero commitment on this street**, never a raise
increment or whole-hand investment. `raise` also represents an opening bet.
A short all-in raise is legal only at the hero's maximum commitment. Calls pay
only the stack available, and uncalled excess is returned independently.
`interface.legalize` normalizes the bot's own intent; the simulator separately
validates it. A failed decision or enforced timeout checks when possible and
otherwise folds. The policy has an internal budget guard that can stop further
value/bluff analysis after it has computed a safe continuation decision.

Construct a fresh Agent or call `reset()` for every table. Anonymized seat
statistics are retained only within a table. The local Table calls each distinct
agent object's optional reset method once before its first deal. The supplied
seed makes strategy mixing reproducible; it never uses another player's cards.

## Strategy and asset provenance

All bot, evaluator, simulator, training and inference code was developed in this
workspace. No other bot, private repository, parent project or team artifact was
read or imported.

- An original bitmask evaluator handles all five-to-seven-card hands. Small
  rank tables are generated from ordinary poker ordering at import.
- The preflop asset has a 169-by-169 all-in equity matrix, with 20,000 trials per
  unordered class pair and separate 20,000-trial random-field equities against
  one through seven opponents. Conditional range weights account for known
  hole-card blockers. Runtime ranges and multiway extrapolation are policy
  approximations, not an eight-player equilibrium.
- A heads-up SB shove/fold versus BB call/fold abstraction was trained with
  alternating CFR+ for 20,000 iterations at effective stacks from 1 to 25 BB.
  Its tiny reported best-response gap applies to that finite abstraction and
  its sampled equity matrix. It does not establish optimal full Hold'em play.
  When enabled, the runtime uses this subgame only in the appropriate short
  heads-up roles. It is disabled in the selected default policy.
- The v2 postflop model was trained on 1.2 million independently generated
  states, with 256 conditional rollout labels per non-exact case. Twenty percent
  structural oversampling improves rare made-hand coverage. Three street-specific
  tree ensembles use 44 exact card features and 14 range/relational features.
- The explicit synthetic range model selects a top preflop fraction, then a
  current-hand-strength fraction with promoted draws and residual weak/bluff
  weight 0.10. Estimated ranges come from observed public actions and heavily
  shrunk within-table statistics. This model is an assumption about opponents.
- The native backend can evaluate the forest directly, enumerate exact heads-up
  river equity against that range model, or run bounded seeded conditional
  rollouts. Impossible or exhausted joint ranges return the model fallback.
  Exactness of enumeration does not make the estimated opponent range exact.
- Postflop decisions combine contestable pot odds, equity realization, position,
  value against stronger continuing ranges, limited bluffs, and all-in risk.
  Important side pots receive a separate equity estimate. Several differently
  capped all-ins still use an approximation for lower pots.
- Supplied payouts enable exact heads-up call ICM comparisons. Open-shove risk
  uses a representative two-player ICM premium and independent caller estimates.
  ICM values of the hypothetical win/fold/loss stack states are exact; tie
  outcomes are approximated using equity, except proven unbeatable hands.
  Full multiplayer ICM game solving and future-game simulation are not claimed.

See `assets/evaluator_NOTES.md` for the generator formats and additional detail.

## Local game assumptions

The complete `RULES.txt` was read. The local harness implements the stated
eight-player start, 2,000-chip stacks, initial 10/20 blinds, 25-hand levels and
200-hand cap. Actual later blind values and table point payouts remain unknown.

Every tournament benchmark explicitly names a **synthetic** payout vector and
blind schedule. The available sensitivity cases are top-three, winner-only,
linear rank points, flat top-three payouts, and moderate/fast/slow blind growth.
None is presented as an official payout or blind schedule.

The simulator uses no antes, TDA-style dead-button progression, heads-up button
as SB, full-raise reopening including cumulative short all-ins, side pots,
uncalled returns, five-card ties, and clockwise odd chips. Simultaneous busts
rank by hand-start stack, with equal stacks sharing occupied place points.
At the hand cap, remaining points are allocated by exact ICM. Button movement,
elimination ties, antes and timing details must be reconciled with the official
simulator once available.

## Reproduction

These commands are intended for a suitable development/compute environment.
Substantial work for this build is run only in this bot's own bounded remote jobs.
Run them from this workspace's root.

```text
python -m unittest discover -s tests -v
python tools/benchmark.py --tables 500 --seed 141000 --field mixed --workers 8 --output measurements/reproduced_development.json
python tools/latency.py --generate --tables 20 --per-street 800 --repeat 10 --output measurements/reproduced_latency.json
```

Linux native build:

```text
g++ -std=c++17 -O3 -fPIC -shared -static-libstdc++ -static-libgcc tools/runtime_equity.cpp -o pokerbot/_native/libb14_equity.so
```

Windows MinGW build:

```text
g++ -std=c++17 -O3 -shared -static -static-libstdc++ -static-libgcc tools/runtime_equity.cpp -o pokerbot/_native/b14_equity.dll
```

Offline asset generation and training need a C++17 compiler, NumPy and
scikit-learn; these training packages are not gameplay dependencies:

```text
mkdir -p build
g++ -std=c++17 -O3 -pthread tools/equity_assets.cpp -o tools/equity_assets
tools/equity_assets --trials 20000 --multiway-trials 20000 --seed 140014 --threads 8 --output assets/preflop.json
python tools/train_pushfold.py --assets assets/preflop.json --iterations 20000 --output assets/pushfold.json --checkpoint .control/pushfold_checkpoint.json
python tools/train_equity.py --prepare-ranks assets/preflop.json --ranks-output build/preflop_ranks.txt
g++ -std=c++17 -O3 -fopenmp tools/postflop_dataset.cpp -o build/postflop_dataset
build/postflop_dataset --rows 1200000 --trials 256 --balanced 1 --seed 140220 --threads 8 --ranks build/preflop_ranks.txt --output build/postflop_train_v2.bin
build/postflop_dataset --rows 24000 --trials 2048 --seed 914007 --threads 8 --ranks build/preflop_ranks.txt --output build/postflop_validation.bin
python tools/train_equity.py --train build/postflop_train_v2.bin --validation build/postflop_validation.bin --derived --iterations 300 --leaves 63 --output assets/equity_model.json --report measurements/equity_validation_v2.json
```

Run the equity generator with `--validate` for exhaustive five-card category
counts. `tools/tune.py` records paired development differences and uncertainty;
it is not a held-out evaluation. Bounded batch scripts in `tools/` record the
actual build sequence. Their remote directory is specific to this bot.

## Evidence and limitations

The selected candidate, held-out results, timing overruns/fallbacks, platform
details, source/asset hashes and experiment boundaries are recorded in
`measurements/README.md` and its linked JSON reports. Development seed sets and
validation labels are kept separate from later untouched evaluation seeds.

Existing mechanics evidence includes exhaustive counts for all 2,598,960
five-card hands, independent evaluator comparisons, C++/Python feature and
forest parity, short-raise and side-pot fixtures, offline-decision checks,
table reset tests, and 23,503 actual-agent stress decisions with zero illegal
outputs or exceptions. That stress is mechanics validation, not strength proof.

No contest hardware or official interface has been tested. Synthetic opponents
are original test policies and are not evidence of performance against submitted
bots. Finite rollouts, range inference, multiway correlation, equity realization,
future tournament play and the unknown scoring structure remain limitations.
There is no claim of absolute optimality or guaranteed contest victory.

## Public references and notices

- [Contest rules source](https://pokerbot.sequoiacap.com/rules.html), with the
  complete supplied text preserved in `RULES.txt`.
- [Poker TDA 2026 rules](https://www.pokertda.com/view-poker-tda-rules/) for
  full-raise reopening, heads-up positions, side pots and dead-button conventions.
- [TDA administrator explanation of short opening all-ins](https://www.pokertda.com/forum/index.php?topic=1456.0)
  cross-checks the distinction between calling a short postflop opening bet
  and making a minimum full raise.
- [CMU's Pluribus research description](https://www.cs.cmu.edu/news/2019/carnegie-mellon-and-facebook-ai-beats-professionals-six-player-poker)
  informed the distinction between an offline policy and expensive live solving.
- [Gilbert, The Independent Chip Model and Risk Aversion](https://arxiv.org/abs/0911.3100)
  provides the ICM recursion and risk-aversion context.
- Compiled libraries use GCC runtime components under the
  [GCC Runtime Library Exception](https://www.gnu.org/licenses/gcc-exception-3.1.html).
  Python and offline training dependencies retain their respective licenses.
