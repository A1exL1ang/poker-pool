# bot_13: independent offline no-limit Hold'em bot

`agent.py` exports `Agent`. This is an independently implemented **local**
interface and laboratory. No official starter kit was supplied, and official
interface compatibility has not been established. Adapt and verify against the
event's actual Agent schema before any submission. Nothing has been submitted,
pushed, or published.

The complete supplied `RULES.txt` was read. Relevant stated constraints are
eight-seat sit-and-gos, 2,000 starting chips, initial 10/20 blinds, 25-hand blind
levels, a 200-hand cap, placement points/ICM, offline decisions within 1 ms,
and a maximum 100 MB submission. The later blind levels and placement points
were not provided. Every benchmark payout vector and later blind level here
is an explicitly **local sensitivity scenario**, not an official contest value.

## Run

Python 3.10+ and the standard library suffice. The stronger native backend is
original C++17, with no third-party poker code or runtime Python dependencies.
Build it ahead of gameplay with a local g++/clang++ compiler:

```sh
python native/build.py
python -m unittest discover -s tests -v
python -m lab.benchmark --tables 64 --seed-start 1000 --lineup mixed --output measurements/example.json
```

The build makes `native/bot13.dll` on Windows or `native/libbot13.so` on Linux.
The prebuilt Windows DLL and Linux library are included in the local gameplay
package, `dist/bot_13.zip`. It includes runtime files and native source; the
workspace additionally holds the laboratory, tests, trainers and measurements.
A failed or unsupported native load uses
a bounded Python heuristic. It does not launch a compiler or run Python Monte
Carlo during a decision. Native initialization and asset loading happen before
play. Gameplay opens no files or sockets, starts no subprocesses, and calls no
online service or AI model.

Use a fresh instance for each table:

```python
from agent import Agent
bot = Agent(seed=13)
decision = bot.act(observation)
# An adapter reusing an instance must call bot.reset() before the next table.
```

`decide` aliases `act`. `reset()` clears all opponent/table state and restores
the initial seed; `reset(seed=...)` also establishes a new initial seed. Seat
statistics are anonymous, local to one table, and never persisted to disk.
With no explicit seed, a fresh instance samples its seed from OS entropy during
initialization. Supplying a seed makes decisions and experiments reproducible.
The selected release parameters load once from `assets/policy.json` if present;
explicit constructor configuration overrides them.

## Local interface and adapter obligations

The complete contract and engine conventions are in [lab/README.md](lab/README.md).
An observation has `hand_id`, `street` (`preflop`, `flop`, `turn`, `river`),
`hero`, `button`, `small_blind`, `big_blind`, `hole_cards`, `board`, `pot`,
`to_call`, `current_bet`, `min_raise_to`, `max_raise_to`, `can_raise`,
`players`, and the complete current-hand `history`. Explicit `big_blind_seat`,
`small_blind_seat` (possibly null), and `seat_count` handle dead buttons/blinds.
`payouts` is optional, ordered from first to last place; missing payouts select
chip EV rather than an invented points distribution.

Each player supplies `seat`, `stack`, `bet`, `committed`, `folded`, `all_in`.
Keep original seats, including eliminated zero-stack seats. Chips in `stack`
exclude all commitments. `bet` is this street's contribution; `committed`
includes every street. `pot` includes **all committed chips and current bets**.
`to_call` is capped to hero's remaining stack. `can_raise` must reflect whether
action is reopened, including cumulative short all-ins.

Cards are integers `4*(rank-2)+suit`, rank 2..14, suits clubs/diamonds/hearts/
spades = 0..3; e.g. 48 = Ac, 51 = As. Canonical two-character strings are also
accepted by the policy. The laboratory always supplies integer cards.

History contains only public decisions, not forced blinds. Events include
`seat`, `street`, `action`, `amount` (new street total), `amount_paid`,
`pot_before`, `to_call_before`, `stack_before`, `current_bet_before`,
`is_all_in`. Complete event metadata gives the best range inference. The
policy does not infer unobserved later actions after it folds. Preflop
frequency denominators count observed opportunities to avoid that censoring
bias. Fold frequency is not estimated from an incomplete act-only stream.

The response is `{"action":"fold"}`, `{"action":"check"}`,
`{"action":"call"}`, or `{"action":"raise","amount":INTEGER}`.
**Raise amounts are raise-to street totals**, including chips already invested
on the current street. Contributions from earlier streets do not enter this amount.
An opening bet uses `raise`. A short all-in below the full minimum is permitted
only at `max_raise_to`. The local adapter clamps strategy proposals and uses
check-when-free/fold-otherwise on an internal error. It cannot preempt an OS
scheduling delay; the laboratory can enforce the stated deadline fallback.
Use `bot13.interface.validate(obs)` while developing an official adapter.

## Strategy

Preflop play combines position/stack-dependent opening ranges, value reraises,
limited blocker bluffs, trained conditional equities, a short-stack shove EV
model, pot odds, realization adjustments, and optional payout-aware risk.
Opponents who have not yet entered a raised pot are treated separately from
already committed contestants, with an uncertainty penalty for later action.

Postflop, a native kernel builds smoothed opponent ranges from public action
sizes, preflop percentiles, board-relative hand strength and private draws.
Fixed-work rollouts estimate equity; heads-up river equity is enumerated
exactly. Shared samples give first-k-opponent shares. Nested side-pot layers
then weight the appropriate eligible contestants and include folded dead
chips. Value bets, semibluffs, limited bluffs, small equity-denial probes, and
stack-to-pot commitment rules use those estimates. Action sampling is local
and reproducible for a seed.

An auxiliary heads-up shove/fold game was also trained, with exact
blocker-conditioned class probabilities and independent best-response checks.
Its certificate applies only to its sampled two-action game, not full NLHE.
The selection evaluation did not favor enabling that component; the selected
release keeps it disabled. Its asset and reproducible trainer remain for review.

When actual payouts are supplied, material calls use exact small-table ICM
for a simplified win/fold/lose counterfactual. Counterfactual chip totals are
conserved. Winner-take-all plus a constant is treated as chip-linear. This is
not a full multiway outcome solver. The native-absent path uses a conservative
large-table risk proxy to keep the Python fallback bounded.

The policy is not a solved or claimed optimal NLHE strategy. Its ranges,
future-call probabilities, realization adjustments and bet sizes are modeling
choices. Future players who can still call are conservatively possible
side-pot contestants without treating their future chips as guaranteed profit.
The native-absent equity fallback is materially weaker, especially against
very tight ranges and multiway opponents.

## Training, evaluation, and timing

`assets/preflop.json` was generated from this bot's own evaluator with seed
130613 and 200,000 rollouts per cell. It covers all 169 starting classes,
uniform 1..7-opponent equity and heads-up equity against top 10/20/35/50/100%
starting ranges. Tied classes are retained whole at a range boundary. Regenerate:

```sh
python tools/train_preflop.py --samples 200000 --seed 130613
python tools/train_pushfold.py --samples 100000 --seed 131313 --threads 8
python tools/experiment.py --plan plans/training_v1.json --workers 8 --output measurements/reproduced_training
python tools/analyze_training.py measurements/reproduced_training
```

Training/tuning seeds are below 600000; separate selection/validation seeds
occupy 600000..899999. Final held-out seeds begin at 900000 and are not
used to choose policy parameters. Experiment plans, per-table outcomes,
source hashes, confidence intervals, deadline counts, and owned job logs are
retained locally. Local baselines include several deliberately exploitable
styles, a uniform-equity reference, same-policy configuration references,
and targeted minraise/wide-shove/premium-nit stress cases. Beating those
opponents does not establish contest ranking or strength against unseen bots.

Initial evidence: the first 5,760-table/90-task tuning batch completed without
task failures; it is training evidence, not held-out performance. The rule
suite includes the entire 2,598,960-hand five-card category histogram,
independent 5/6/7-card comparisons, blocker-conditioned sampling, ICM, side
pots/refunds, ties/odd chips, minimum raises/reopening, short blinds, dead
buttons, heads-up transitions, observation privacy, and chip conservation.

The 1 ms limit applies here to the warmed `act()` callback. Import, constructor,
asset loading, process startup and JSON transport are outside that timer and
must be reconciled with the official host. Fixed sample counts bound algorithmic
work but do not guarantee hard real-time execution on a shared operating
system. Early Windows training replay measured 17,931 calls, median 220.9 us,
p99 618.3 us, and 32 overruns, with zero internal/native errors. The first
Linux training base configuration recorded 42 deadline fallbacks in 59,683
decisions across five scenarios, with zero action errors. These are diagnostics
of earlier checkpoints, not release guarantees. Frozen release measurements
follow below.

## Frozen release evidence

`assets/policy.json` records the selected `accurate` configuration. It uses
192 equity samples heads-up, 96 against two or three opponents, and 32 in
larger pots. Runtime source and asset hashes match the frozen experiment;
see [measurements/FINAL.json](measurements/FINAL.json). Parameters were chosen
before held-out outcomes were inspected and were not changed afterward.

The final controlled batch ran 11,520 held-out tables: 9,216 using the release,
2,048 paired baseline tables, and 256 explicit native-absent fallback tables.
Each release scenario had 1,024 tables. The following means and unadjusted
normal 95% intervals refer only to declared LOCAL scenarios:

- Mixed styles: 29.11 points, interval 27.86–30.37.
- Mixed stress styles: 20.46 points, interval 19.23–21.69.
- Minraise pressure: 11.36 points, interval 10.29–12.43.
- Uniform-equity opponents with doubling blinds: 28.64 points, interval 27.27–30.02.
- Base-policy configuration references: 12.35 points, interval 11.23–13.47.
- Pressure-policy configuration references: 12.73 points, interval 11.62–13.83.
- Previously unused reference configuration with doubling blinds: 13.19 points,
  interval 12.08–14.30.
- Mixed styles with the illustrative 7/6/5/4/3/2/1/0 ladder: 5.58 points,
  interval 5.48–5.68, from a 28-point pool.
- Tight opponents, scored by chips: 6,163.5 terminal chips, interval 5,700.0–6,627.0.

The first seven scenarios use the illustrative 50/30/20/0/0/0/0/0 vector,
whose equal-seat reference is 12.5 points. Minraise pressure remains a
measured weakness. The same-policy reference intervals overlap 12.5; these
results do not demonstrate superiority to every reference, let alone contest
opponents. The baselines and scenario mixture are not the official field.

The release made 1,661,888 decisions in those tables, with zero internal
fallbacks, native errors or invalid/exception actions. Deadline enforcement
caused **139 timeout fallbacks (0.008364%)**. Per-scenario p99 callback times
were 0.353–0.400 ms. The largest observed parallel-harness delay was 197.5 ms;
these are wall-clock measurements, including possible scheduling, GC and
harness effects. No causal attribution or hard real-time guarantee is claimed.

A separate serial replay made 32,270 calls over 6,454 saved decisions,
repeated five times: median 0.161 ms, p99 0.336 ms, maximum 0.987 ms, zero
overruns and zero internal/native failures. A synthetic concentrated-range/ICM
stress test made 16,000 full-callback calls, with zero illegal actions or
internal/native failures and one 1.813 ms overrun. Its largest case p99 was
0.455 ms. These timing exercises are not independent playing-strength evidence.

The final Linux audit ran 64 tests: 62 passed, zero failures/errors and two
optional artifact-dependent skips (old training corpus and offline toy
executable). Earlier checks exercised both skipped artifacts; the final job
also generated and replayed a new observation corpus. Isolated package
validation checked every manifest hash, loaded the bundled native library,
verified safe actions/reset, and blocked files/sockets during decisions.
The isolated import plus initialization took 0.703 seconds on that run,
**outside** the callback timer. Startup/transport timing must be reconciled
with the actual official host before claiming compatibility.

Reproduce the frozen scenarios on an appropriately provisioned machine:

```sh
python tools/experiment.py --plan plans/heldout_v4.json --workers 8 --output measurements/reproduced_heldout
python tools/package.py
python tools/validate_package.py --output measurements/package_validation.json
```

Seeds and policy randomness are reproducible. Wall-clock timeout decisions
can change game trajectories across machines, so deadline-enforced scores
need not match bit for bit. Current artifacts retain the actual observed
outcomes. Refinement stopped after 21,120 controlled tuning tables and 24,576
separate selection tables: added shove/fold and broader-range variants did
not improve selection evidence, and the remaining reference differences
have substantial uncertainty. Further claims require the official interface,
hardware, payouts and independently evaluated opponent field.

Replay timing separately from playing-strength evaluation:

```sh
python tools/latency.py --corpus measurements/training_observations.jsonl.gz --repeats 3 --output measurements/replay.json
python -m lab.benchmark --tables 128 --seed-start 1000 --lineup minpressure --enforce-deadline --output measurements/deadline_example.json
```

The default replay leaves Python GC enabled. The optional `--disable-gc`
setting is an explicit diagnostic, not a hidden production setting.

## Provenance and remaining assumptions

All bot, evaluator, simulator and baseline code was authored in this isolated
workspace. No other team's code, artifacts, messages, private repositories or
experiments were consulted. Public research informed the choice of small
action abstractions and empirical evaluation, rather than a full-game search
inside a 1 ms Python callback: [Brown and Sandholm, 2019](https://noambrown.github.io/papers/19-Science-Superhuman.pdf).
Payout-dependent risk motivates explicit ICM outcomes: [Gilbert, 2009](https://arxiv.org/abs/0911.3100).
The simulator's standard-rule references and explicit tie/blind conventions
are documented separately in the laboratory README.

Only newly requested, bounded `bot_13` jobs in the authorized private remote
directory were used. Their ledger is `.control/jobs.json`; pre-existing ORCD
jobs and the shared SSH master were never attached to or modified. Remote
training is a development step only. The collected gameplay package needs no
cluster, credentials, network, compiler, trainer, or external service.
