# Reproducing the equity experiment

Training is separate from live gameplay. C++17, OpenMP, Python 3, NumPy, and
scikit-learn 1.5 were used for generation/fitting. The generated model uses
only standard-library Python at runtime. No external poker code or hand-range
database was copied.

```sh
g++ -O3 -std=c++17 -fopenmp training/generate.cpp -o training/generate
training/generate --mode postflop --rows 400000 --trials 384 --threads 8 --seed 180701 --output training/train.csv
training/generate --mode postflop --rows 18000 --trials 4096 --threads 8 --seed 189913 --output training/heldout.csv
training/generate --mode preflop --trials 40000 --threads 8 --seed 181643 --output training/preflop.csv
training/generate --mode premium --trials 80000 --threads 8 --seed 181911 --output training/premium.csv
OMP_NUM_THREADS=8 python3 training/fit.py --train training/train.csv --heldout training/heldout.csv --preflop training/preflop.csv --premium training/premium.csv --iterations 180
python3 training/compile_model.py --model equity/model.json --output equity/compiled.py
python3 -m unittest discover -s tests -p 'test_equity*.py' -v
python3 training/validate.py --native training/generate --samples 20000
```

Generation is deterministic by row seed regardless of OpenMP scheduling.
Training/evaluation states have distinct seeds. Each postflop row uniformly
chooses flop/turn/river and one of five ranges; opponent counts 1,2,3,5,7 have
probabilities 60%,20%,10%,5%,5%. The 4/6-opponent predictions interpolate between
neighboring tree splits rather than having direct postflop training samples.
Preflop includes all 1–7 counts. Cards and completed boards are otherwise
uniform subject to blockers and the specified opponent range. See the equity
README for the multiway range-sampling limitation.

`fit.py` emits `equity/model.json`, `equity/preflop.json`, and a metrics report
with training-sample and held-out errors, stratified errors, largest errors,
and native-scikit-learn versus exported-Python prediction equality. The
held-out set has 4096 Monte Carlo trials per row, so targets still have sampling
noise (worst-case standard error about 0.0078 for heads-up outcomes).

The validation script independently checks 200 exactly enumerated river
equities against uniformly random single opponents, measures cold-per-state
live prediction latency, and counts wall-clock calls above one millisecond.
These are component measurements, not full bot decision latency. Operating
system scheduling can produce isolated wall-clock overruns even when CPU
work is much smaller. Full-agent measurements belong to the head's benchmark.

If a held-out report is used to select/refine a model, it becomes development
evidence. Generate a fresh independent final holdout for final evaluation and
label the reports accordingly. The initial untrained validation report is
retained only to document the checkpoint and must not be presented as trained
model performance.

The original `heldout.csv` was inspected while refining the model and is now
development validation. V2 fits separate street models, using the same
training data and frozen 75-feature encoder:

```sh
OMP_NUM_THREADS=8 python3 training/fit.py --train training/train.csv --heldout training/heldout.csv --street-models --iterations 300 --leaves 63 --depth 9 --output equity/model_v2.json --compiled-output equity/compiled_v2.py --report training/metrics_v2.json
```

V3 adds independently generated rare-texture training examples at half weight.
These oversample board quads, full houses, trips, two pairs, monotone boards,
four-flush and straight textures, often giving hero an improvement card:

```sh
training/generate --mode stress --rows 120000 --trials 512 --threads 8 --seed 182311 --output training/stress_train.csv
OMP_NUM_THREADS=8 python3 training/fit.py --train training/train.csv --extra-train training/stress_train.csv --extra-weight .5 --heldout training/heldout.csv --street-models --iterations 300 --leaves 63 --depth 9 --output equity/model_v3.json --compiled-output equity/compiled_v3.py --report training/metrics_v3.json
```

The generated modules' exact equality to the scikit-learn model is checked on
512 validation rows during fitting and on 500 additional random states by
unit tests. `training/stress.py` is a separate hand-picked development test,
using exact single-opponent river enumeration. Its distribution is deliberately
unrepresentative and must not be reported as a tournament performance estimate.

After freezing the final assets/source, the head can generate an independent
test with uniform opponent counts 1–7 (including unseen 4/6-opponent cases):

```sh
training/generate --mode final --rows 28000 --trials 8192 --threads 8 --seed 1890017 --output training/final_test.csv
python3 training/evaluate_final.py --data training/final_test.csv --seed 1890017 --label-trials 8192 --expected-model-sha FROZEN_SHA256 --output training/final_evaluation.json
```

`evaluate_final.py` checks the frozen model hash before reading final labels
and calls the actual runtime including exact rare-board overrides. It reports
errors by street/range/opponent count, component wall-clock and CPU timing,
model/data hashes, and Monte Carlo label uncertainty. It does not fit or
select a model. The final test is not run by the equity helper before freeze.
On this native Windows Python build the thread CPU clock advanced in roughly
15.625 ms increments; its per-call CPU percentiles and overrun counts are
therefore unusable. Wall-clock timings remain reported, and Linux CPU timing
is separately measured. The scripts flag visibly quantized CPU timing.
