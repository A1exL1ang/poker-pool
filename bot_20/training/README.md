# Independent equity training

Gameplay uses Python's standard library only. Training additionally needs a
C++17 compiler, NumPy, and scikit-learn. OpenMP parallelizes independent sample
rows; choose at most eight threads in this team's authorized compute job.

Reproduce the selected model with `bash training/run_selected.sh` from the
workspace, inside this bot's own bounded compute allocation. It fixes the
preflop ordering seed at 103117 (30,000 trials per class), refined preflop table
seed at 327119 (12,000 trials per cell), selected training seed at 471029
(1,000,000 states with 512 trials each), and validation seed at 781237 (10,000
states with 1,024 trials each). It fits the `street48` configuration: one model
per street, 48 trees each, at most 511 leaves and depth 16, learning rate 0.22,
and feature version 2 with 90 public-card features. It writes a separate
reproduction report and then installs the selected asset. It never generates
or reads final evaluation data. Source/library versions and data hashes matter
for bitwise reproduction; the measured training environment used Python 3.12.7,
NumPy 1.26.4, and scikit-learn 1.5.1.

The earlier `bash training/run_training.sh` command creates a preflop
baseline table and a compact model in `pokerbot/assets`. This baseline command
does not reproduce later selected candidates; their exact configurations and
input hashes are recorded in `measurements/equity_experiments*.json` and each
model's metadata. No code downloads or online calls
occur. The two training seeds and validation seed are distinct. The validation
file is available for model selection; it is **not** held-out final evidence.

After freezing the model, create and evaluate the final set:

```sh
training/generate --mode postflop --order training/data/order.csv --rows 12000 --samples 2048 --seed 991237 --out training/data/final_test.npy
python3 training/evaluate_equity.py --test training/data/final_test.npy
training/generate --mode preflop_test --order training/data/order.csv --rows 3000 --samples 4096 --seed 110327 --out training/data/preflop_test.npy
python3 training/evaluate_preflop.py --test training/data/preflop_test.npy
```

Candidate selection uses `training/experiment_equity.py`, which fits the
predeclared universal and per-street tree configurations on the supplied
training and validation files. It writes candidate assets separately and does
not replace the runnable default. Feature version 2 adds explicit relations
between the made hand and board: higher possible straights, one-card straight
threats, board kicker differences, and higher full-house threats. The runtime
remains compatible with feature-version-1 models through its 64-feature prefix.

Preflop ordering is the average showdown share of each of 169 classes against a
uniform random hand, estimated from 30,000 independent runouts per class. Class
indices use a 13 by 13 ascending-rank matrix: below-diagonal suited, above-
diagonal offsuit, and paired diagonal. Combination weights (4, 12, and 6) turn
the ordering into percentiles. The original table used eight range fractions;
the refined generator uses 18 fractions from 0.02 to 1 and seven opponent counts,
including ties. Fraction interpolation is
logarithmic. Its Monte Carlo standard errors are stored alongside the cells.
The runtime respects each table's minimum preflop fraction; postflop queries
remain within the trained 0.05-to-1 range. The final table can be generated with
`--mode preflop --samples 12000 --seed 327119` using the fixed ordering. A separate
`--mode preflop_test --rows 3000 --samples 4096 --seed 110327` samples continuous
fractions for evaluation with `training/evaluate_preflop.py`.

A modeled opponent's preflop weight is 1 inside the chosen top-fraction range
and `0.02 * exp(-6 * (percentile - fraction))` outside it. The full-range setting
is uniform. The small tail ensures support after blockers even in multiway
pots. This is a deliberately documented range family, not a claim about actual
opponents or equilibrium ranges.

Postflop pressure interpolates between that preflop range and a range weighted
by `0.02 + 3 * current_hand_percentile**6 + draw_quality`. Current hand percentile
comes from exact present-street ranks of all two-card holdings consistent with
the board. Draw quality gives additional weight to private-card-assisted flush
and straight draws. No future board card enters these weights. Each simulated
opponent hand is sampled from this range conditioned on the cards already
drawn; this sequential conditional sampling is an approximation to jointly
conditioning independent ranges on disjoint hands. We then draw the remaining
board uniformly and record the hero's exact pot share, including multiway ties.

Features use only the hero's hole cards and public board, with suit and hole-
order invariance. The model additionally receives opponent count, range
fraction, pressure, and hero preflop percentile. Exported trees use ordinary
numeric comparisons and floats; their predictions are checked against the
training library before export is accepted. At initialization, the runtime
validates numeric tree fields and compiles fixed-syntax conditional expressions
from them; no observation text or source strings from model fields are
executed. The compiled immutable predictors are cached across agent instances.
This removes per-node interpreter-loop overhead during timed actions. A model/schema failure exposes
`model_loaded=False` and retains an analytic fallback for interface bring-up.
The final bot should package both trained assets and verify those flags.

The final equity test measures approximation of this specific range model,
not playing strength. Its labels retain Monte Carlo uncertainty, reported as
an estimated noise floor. Tournament tests must independently evaluate policy
quality, payout sensitivity, opponent adaptation, and 1 ms action latency.
