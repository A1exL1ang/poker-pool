# Offline equity assets

All evaluation, feature extraction, data generation and inference code here is
original to this bot. No external poker code or private repository was used.
The scientific Python dependencies are used only to fit models offline.
Gameplay imports only Python's standard library and the generated assets.

## Encoding and target

Card id is `4 * (rank - 2) + suit`, suits clubs/diamonds/hearts/spades.
`equity.evaluate` computes the exact best five cards from five, six or seven
cards, including wheel straights, two trips, three pairs, and board ties.
Scores increase with strength; `score >> 20` is category 0 through 8.

The learned postflop targets are heads-up showdown equity with uniformly
sampled future runouts against (a) all legal opponent combinations and (b) a
fixed top 20% preflop range. Ties receive half a win. A fixed range is **not**
a posterior range conditioned on the opponent's betting actions. The strategy
layer decides how to use the estimates and handles multiway play separately.
Three analytically locked river board families return exactly 0.5 HU equity.

The 169 preflop classes have separate equity estimates against top
5%, 10%, 20%, 35%, 50%, and 100% ranges, weighted by physical combo count.
Range order is estimated in an earlier independent Monte Carlo experiment
against uniformly random hands. The exact threshold can include an entire
hand class crossing the nominal percentage; order values in `preflop.json`
make that definition explicit. Opponent sampling excludes every known card.
An additional asset provides nominal top 1% and 2.5% curves through
`preflop_narrow_equities`, preserving the six-range API. Whole class boundaries
make their actual fractions 1.3575% and 2.7149%; metadata states this directly.

The model is unnecessary on the river once `USE_EXACT_RIVER` is enabled.
`exact_river` aggregates all legal opponent hands by their 91 rank pairs,
uses a 49,205-entry original suit-free rank table, then corrects combinations
that complete a flush. A second card outside the dominant board suit is
grouped by rank with exact suit blockers, keeping the correction count small.
This is exact heads-up enumeration for both fixed ranges and includes exact
tie probabilities. The model remains available for reference ablations.
The proof relies on a river board having at most one suit with three or more
cards; every possible flush must use that suit. Dry boards need no correction.
Tests compare the aggregation against all 990 physical opponent combinations
on randomized boards in every suit family and targeted straight-flush/quads
cases. The compressed rank table is 143,153 bytes, built by original code.

## Reproduce

Build with C++17, then run bounded offline jobs (example small checkpoint):

```text
g++ -O3 -std=c++17 -o training/generate training/generate.cpp
training/generate order training/order.txt 30000 1200101
training/generate preflop training/preflop_initial.csv 20000 1200103 training/order.txt
training/generate postflop training/train_initial.csv 256 1200104 training/order.txt 30000
training/generate postflop training/heldout_initial.csv 1024 1200105 training/order.txt 2000
python training/train_model.py --train training/train_initial.csv --test training/heldout_initial.csv --preflop training/preflop_initial.csv --order training/order.txt --iterations 100
```

Windows executables use `.exe`. Fitting requires numpy and scikit-learn, with
OpenMP/OpenBLAS capped to four threads by the script unless explicitly set.
HistGradientBoostingRegressor models are exported to compact JSON and an
equivalent ordinary Python expression forest. No pickle or sklearn import is
used during gameplay. `assets/preflop.json` and `assets/equity_predict.py` are
the gameplay assets; the JSON forest is a readable inference fallback.

## Final production reproduction

The larger experiment uses 300,000 training situations with 1024 simulations
for each target and 15,000 separate held-out situations with 4096 simulations
per target. Preflop order uses 200,000 simulations/class, then the six range
curves use 150,000/class/range. The following script uses at most eight
single-thread workers; run it only in a new bounded compute allocation, never
on a login node or another allocation. Data generation is deterministic for
the recorded source, seeds, and commands.

```text
bash training/reproduce_data.sh
python training/export_preflop.py training/data/preflop.csv training/data/order.txt
training/generate narrow training/preflop_narrow.csv 200000 1200501 training/data/order.txt
python training/export_narrow.py training/preflop_narrow.csv
python training/build_rank_table.py
python training/train_joint.py --train training/data/train0.csv training/data/train1.csv training/data/train2.csv training/data/train3.csv training/data/train4.csv --validation training/data/train5.csv --leaves 127 --depth 10 --iterations 140 --output-dir .control/joint_reproduction
python training/finalize_joint.py --train training/data/train0.csv training/data/train1.csv training/data/train2.csv training/data/train3.csv training/data/train4.csv training/data/train5.csv --heldout training/data/heldout.csv
```

The final forest uses joint residual boosting: 140 original fitted regression
trees, each with two output values per leaf, maximum 127 leaves/depth10, learning
rate0.1, and 80,000 sampled training rows per stage. Each tree is traversed once
for both range targets. This reduced runtime relative to the full 400-tree
reference with a small validation error tradeoff. The generated source is an
ordinary Python conditional expression forest; no sklearn or binary native
dependency is present during gameplay. The full reference model/report is
preserved separately, and final selection uses only training0..4 versus
training5 validation. Both model architectures evaluate the independently
seeded held-out set once per final model; those results do not select the
joint model or its settings.

`training/model_report.json` records the final forest's held-out errors and
data sources; these estimates include Monte Carlo label noise, whose
4096-sample maximum standard error is about 0.78 percentage points. The forest
river errors are reference measurements: production river decisions bypass
the model and use exact enumeration. The held-out set is never fitted.

## Verification

```text
python -m unittest discover -s tests -p test_equity.py -v
python training/validate_equity.py --exhaustive-five
python training/validate_equity.py --rivers 1000 --output training/exact_river_report.json
python -m unittest discover -s tests -p test_dry_river.py -v
```

The tests compare the fast 5/6/7-card evaluator with an independently written
five-card sorting/counting evaluator and exhaustive best-five subset search.
The exhaustive option checks all 2,598,960 five-card hands against the
combinatorial category counts. The diagnostic river benchmark enumerates all
990 legal opponent hands, avoiding Monte Carlo label error. This additional
diagnostic set is distinct from the model fitting data and held-out labels.
Suit relabeling and hole-card order must leave the feature vector and equity
unchanged. Draw features require a hole card to contribute to the draw.

Limitations: equity models are approximations, not solved betting policies;
rare texture estimates can have much larger error than average. Preflop range
order has sampling noise. Heads-up equity cannot by itself represent all
multiway correlation, side-pot exposure, or split-pot equity. Neither this
training environment nor its schema claims official interface compatibility.
