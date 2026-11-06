# Equity subsystem provenance and reproduction

All evaluator, generation, feature, export and deployment code in this subsystem was written independently in this workspace. Runtime imports only the standard library. Training uses GCC C++17 and scikit-learn/NumPy; neither is a deployment dependency.

`bot.evaluator.evaluate` ranks standard Hold'em hands using all five-card subsets implicitly through rank/suit histograms. Larger integers win, and category is integer division by 759375. It supports 5–7 cards; internally the equity feature extractor also uses its partial-hand descriptions for three/four-card boards. Inputs must be valid distinct cards using rank `card // 4`, suit `card % 4`.

The preflop table ranks all 169 starting classes by 16,000 simulated random-opponent runouts per class. It then simulates 16,000 runouts for each class against each of eight combo-weighted top ranges (3%, 7%, 12%, 20%, 35%, 50%, 75%, 100%). Card removal is respected. These are internally defined equity-ranked ranges, not GTO opening ranges. Similar classes can swap order due to Monte Carlo noise. These outputs estimate heads-up equity; multiway equity and betting-conditioned ranges require additional strategy assumptions.

The postflop model fits four equities (versus 7%, 20%, 50%, 100% starting ranges). Inputs comprise 41 integer features of both hole cards and the flop/turn/river. Training/validation/test generator seeds are disjoint. Model selection uses validation mean absolute error. The test set is only used for the selected model's final error report. Labels are Monte Carlo estimates; error metrics include their sampling noise. The model does not observe opponent betting, position or stacks, and should not be called an equilibrium solver.

Reproduction on Linux, from workspace root:

```sh
g++ -O3 -std=c++17 training/native_equity.cpp -o training/native_equity
training/native_equity preflop artifacts/equity_preflop.json 16000 151515
training/native_equity data artifacts/equity_train.bin 240000 15101 192
training/native_equity data artifacts/equity_valid.bin 12000 15102 1024
training/native_equity data artifacts/equity_test.bin 12000 15103 1024
python training/equity_train.py --train artifacts/equity_train.bin --valid artifacts/equity_valid.bin --test artifacts/equity_test.bin
cp artifacts/equity_model.json.gz artifacts/equity_model_full24.json.gz
python training/equity_optimize.py
python training/equity_finalize.py --trees 8
python training/equity_audit.py --data artifacts/equity_test.bin
python -m unittest discover -s tests -p test_equity.py -v
g++ -O3 -std=c++17 training/native_validate.cpp -o training/native_validate
training/native_validate
python -m training.verify_native_features
```

The exact preflop asset can differ slightly between standard library implementations because the seeded generator's implementation details are not promised across compiler platforms; the included asset is the reproducible best checkpoint. `equity_train.py` exports numeric JSON arrays compressed with gzip. Deployment does not deserialize executable pickle objects or compile native code.

The completed experiment used **240,000** training rows with 192 sampled runouts per range, 12,000 validation and 12,000 held-out test rows with 1,024 runouts per range. Validation plus latency selected the first **8 trees** of the fitted 24-tree forest. Validation MAE was 0.0172862 for eight versus 0.0164400 for 24. Runtime stores nested immutable tuples to reduce pointer lookups and completes bounded garbage-collector untracking during initialization. The eight-tree gameplay asset is 5,160,408 bytes; the original 24-tree artifact is retained separately for reproduction and excluded from gameplay packaging. Final held-out deployment MAE is 0.0172640 and RMSE 0.0259041. In one 12,000-call Windows run, inference median was 82.5 microseconds, p99 172.2 microseconds, maximum 899.1 microseconds, with zero over one millisecond. These figures measure equity inference alone, exclude loading, and are not a hard latency guarantee or a full-agent benchmark.

Validation already completed before training: all 2,598,960 five-card hands match an independent five-card reference; 10,000 random seven-card hands match exhaustive subset enumeration; 19,664 native/Python feature vectors match, including all 13 possible turn-board quads with every legal hole pair. The seven-card Python evaluator also passes 3,600 independent randomized 5–7-card tests. `training/equity_audit.py` verifies deployed feature parity, held-out equity error, and wall-clock inference latency. See generated metrics for measured numbers rather than treating these checks as evidence of contest performance.

Runtime files are `bot/equity.py`, `bot/evaluator.py`, `artifacts/equity_preflop.json`, and `artifacts/equity_model.json.gz`. Call `require_assets()` during construction, outside decision timing. The agent should reset all strategy/opponent state between tables; equity assets contain no table state. Missing-asset developmental estimates exist only to retain a runnable build checkpoint and must be rejected by final-package construction.
