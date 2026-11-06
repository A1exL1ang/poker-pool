# Independent evaluator and preflop labels

All source in `pokerbot/evaluator.py`, `pokerbot/equity.py`,
`tools/holdem.hpp`, and `tools/equity_assets.cpp` was developed independently
for this bot. It uses no third-party poker implementation or poker assets.

Cards are integers `4 * (rank - 2) + suit`, with rank 2..14 and suit 0..3.
The Python and C++ evaluators produce identical integer scores. Higher wins;
the category is `score >> 20`, high card=0 through straight flush=8. The
remaining bits encode the five relevant kickers using four bits per rank.
The fast path assumes valid unique five-to-seven-card input. The checked
Python function validates these assumptions.

`feature_vector(hole, board)` returns 44 suit-invariant integer features.
The authoritative ordered schema is `pokerbot.equity.FEATURE_NAMES`, also
available as `holdem::kFeatureNames`. They describe made hand and kickers,
hole cards, board ranks/multiplicities, suit texture and blockers, straight
potential, and the board's own made hand on the river. `hand_features` is a
readable dictionary with the same fields. These features themselves are not
an equity estimator or a game-theory solver.

Reproduce the offline preflop asset with:

```sh
g++ -O3 -std=c++17 -pthread tools/equity_assets.cpp -o tools/equity_assets
tools/equity_assets --trials 20000 --multiway-trials 20000 --seed 140014 --threads 8 --output assets/preflop.json
```

This is offline training, not timed gameplay. All 169 classes are ordered in
a conventional 13-by-13 grid with rows/columns A..2, upper triangle suited,
lower triangle offsuit. JSON contains explicit class names, combo counts,
metadata, a 169-by-169 heads-up showdown equity matrix, and 169-by-7 random
opponent showdown equities. Heads-up labels sample actual legal card
combinations within each class uniformly and sample uniform legal runouts.
Only one direction of each class pair is simulated; the opposite is its
complement. Identical-class equity is exactly 0.5 by player-exchange
symmetry. Monte Carlo error remains; these are not exact enumerations.
Multiway entries are expected pot shares including tied winners, against
one through seven uniformly random opponents. They reuse a runout and an
ordered sequence of opponents, so estimates across opponent counts are
correlated. Task-local seeds make label values independent of thread count.

Validation commands:

```sh
python -m unittest discover -s tests -p 'test_evaluator.py' -v
python -m unittest discover -s tests -p 'test_equity.py' -v
tools/equity_assets --validate
```

If the C++ generator executable is present, `test_equity.py` also checks
1000 random Python/C++ evaluator and feature vectors. The optional `--probe`
mode supports differential testing via stdin. The `--validate` mode counts
all 2,598,960 five-card hands and compares the nine category totals with
standard combinatorial counts.

`evaluator_preflop_smoke.json`, if present, was made with only 50 samples per
class pair and 100 multiway samples as a format/semantics smoke test. It is
not a training asset and must not be used by the candidate.

## Heads-up shove/fold refinement

`tools/train_pushfold.py` trains a restricted two-player zero-sum game using
the preflop matrix. The small blind can shove or fold, and the big blind can
call or fold. Both start with the given effective stack including the posted
0.5/1 blinds; there are no antes. The SB fold payoff is -0.5 BB, the BB fold
payoff to SB is +1 BB, and a called shove has SB payoff
`(2 * equity - 1) * effective_stack_bb`.

The trainer enumerates exact conditional villain-class combo counts after a
canonical hero hand. Each row sums to `C(50,2)=1225`, and the physical deal
weights obey exact player-exchange detailed balance. It uses alternating
CFR+ updates and linearly weighted average strategies. Every checkpoint
writes a resumable training state to `.control/pushfold_checkpoint.json` and
a runnable best candidate to `assets/pushfold.json`. The output records
per-depth best-response gains, their sum (NashConv gap), and half that gap,
all in BB per hand, plus a convergence trace and the source preflop hash.

```sh
python tools/train_pushfold.py --assets assets/preflop.json --output assets/pushfold.json --iterations 20000
python -m unittest discover -s tests -p 'test_pushfold.py' -v
```

NumPy is required for offline training only; its process is restricted to one
BLAS/OpenMP thread. `--validate-only` validates assets, exact combo counts and
simple payoff toys without NumPy. The saved strategies solve only this
shove/fold abstraction. A small measured gap does not establish exploitability
in full no-limit Hold'em, and showdown labels retain Monte Carlo error.

## Optional native runtime equity backend

`tools/runtime_equity.cpp` provides a C ABI consumed only through standard
library `ctypes` in `pokerbot/native.py`. It does not compile, install, spawn
processes or call any network service during gameplay. Missing/incompatible
binaries cause `NativeEquity.available=False` and callers retain the portable
Python model. Candidate packaged paths are `pokerbot/_native/b14_equity.dll`
on Windows and `pokerbot/_native/libb14_equity.so` on Linux. These builds have
platform/runtime requirements; availability is detected, never assumed.

```sh
# Linux, without host-specific CPU instructions:
g++ -O3 -std=c++17 -fPIC -shared tools/runtime_equity.cpp -o pokerbot/_native/libb14_equity.so
# MinGW Windows, including static GCC/C++/pthread support libraries:
g++ -O3 -std=c++17 -shared -static -static-libgcc -static-libstdc++ tools/runtime_equity.cpp -o pokerbot/_native/b14_equity.dll
python -m unittest discover -s tests -p 'test_native.py' -v
```

The backend initializes from the same 169 preflop percentiles as the offline
dataset generator. Its range is top-q preflop classes, then current-hand
quantile top-m with residual 0.1 weight. Flop/turn draws receive the same
explicit promotions as the dataset labels. Heads-up river pot share is
enumerated exactly under this **synthetic range**, including ties. It is not
exact equity against the opponent's unknown actual strategy.

Other streets and multiway holdings use seeded Monte Carlo with joint
rejection, conditioning the product of the selected ranges on card-disjoint
hands. Work is bounded by requested samples (1..2048) and an aggregate
rejection cap. An impossible or excessively concentrated range returns
`None` for the Python fallback; it never silently switches to biased
sequential conditioning. `river_equities` shares preparation for up to eight
HU river m cutoffs. `river_iid` is separately named because it approximates
multiway river equity with independent opponent rank outcomes and ignores
card correlation; it is not the same as card-disjoint Monte Carlo.

Unit evidence includes independent Python river enumeration with uniform and
weighted ranges, deterministic rollouts, batch parity, invalid/impossible
range fallbacks and eight-way royal-board tie shares. The tiny Windows
development timing smoke is recorded in `.control/native_smoke_windows.json`;
it does not replace whole-agent latency validation on the evaluation host.

The native backend also accepts copied offline surrogate trees at startup:
`configure_model(MODEL.models, MODEL.feature_count)`. Predictions use
`predict_model(features, q, m, opponents)`, accepting raw 44 features or an
already augmented model vector. Both the 47-feature first model and the
58-feature derived model are supported. Trees are validated for feature/child
bounds, cycles, finite values, and bounded depth before use. Static native
storage supports up to 131072 nodes and 2048 trees per street, with at most
128 feature columns and 128 branch steps per tree; inference allocates no
native heap memory. Missing optional model functions on an earlier binary
return `False` at configuration and leave ordinary equity calls usable.

Rank percentiles and surrogate arrays are **process-global startup data** in
the native library. A deployed process configures its one consistent asset
set before gameplay and does not mutate it online. Synthetic unit fixtures
restore the actual `TABLES.percentile` and `MODEL` afterward to avoid shared
process contamination. Native prediction preserves Python tree summation and
clipping semantics; 1000 random real-asset predictions matched Python to 14
decimal places, and a synthetic 58-feature model exercises derived features.
Tiny 100-call v1 timing is saved in
`.control/native_model_smoke_windows.json`; this is development timing only.
