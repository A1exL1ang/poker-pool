# Equity subsystem

All live code is Python standard library only. Import `equity.model.estimate`:

```python
estimate(hole, board, range_strength=0.25, opponents=2)
```

Cards use `card = 4 * (rank - 2) + suit`, ranks 2–14, suits 0–3. The return
value is expected showdown pot share, including split pots. The range argument
is the top fraction of the fixed combination-weighted ordering in `preflop.py`;
0.06 means tight and 1.0 means every legal combination. This ordering is a local
abstraction, not a Nash solution or an official contest chart. All combinations
in a boundary hand class are included, so actual fractions can slightly exceed
the named cutoffs. Opponent count is clamped to 1–7.

`describe(hole, board)` caches the 75 deterministic features and
`estimate_features(descriptor, range_strength, opponents)` reuses them. Feature
7 is the hero's made-hand category (0 high card through 8 straight flush),
8–12 its tiebreak ranks, 13 the board category, 14–18 board tiebreak ranks,
19–23 board ranks, 31–32 board card counts matching each hole suit, 33 the
maximum combined suit count, and 41–48 straight/draw features. See source for
the full exact schema. Hand categories on a three/four-card board are partial
feature summaries, not complete five-card hands.

`preflop_equity` uses Monte Carlo tables for 169 classes, five opponent ranges,
and seven opponent counts; range fractions between trained endpoints are
linearly interpolated. Postflop predictions use exported histogram gradient
boosting trees with range fraction and opponent count as two extra features.
The stronger v2/v3 candidates use one separate model per street and evaluate
only that street's trees. `compiled.py` is generated ordinary Python with
conditional expressions; its SHA256 binding to `model.json` is checked once
at import. This removes per-node interpreter loops without a native runtime
dependency. The interpreted representation remains a verified fallback.
Opponent count has a monotonic decreasing constraint. The model was trained
on true multiway simulated pot share, not a heads-up-equity power heuristic.
Range interpolation in the tree model is piecewise constant across learned
thresholds. No online sampling, training, service, or network call occurs.

The separate `estimate_premium(hole, nutted=False)` heads-up table models
QQ+/AK; `nutted=True` models KK+. These extra ranges do not change the five
postflop model range fractions. `forced_tie(hole, board)` and
`forced_tie_features(descriptor)` identify universal split-pot boards so the
strategy can avoid distorting those exact shares with a pressure adjustment.

The generator samples opponent ranges sequentially without replacement and
then uniformly completes the board. With multiple constrained ranges this is
an approximation to uniformly sampling all jointly compatible assignments:
earlier sampled opponents can affect later opponents' available combinations.
It is exact for one opponent and for unrestricted ranges. All opponents share
one range parameter, which is another abstraction. The model does not infer
postflop selection from betting history: the strategy layer must account for
the tendency of strong hands to continue betting and calling.

Royal flushes, un-improvable quads, and un-improvable broadway straights on the
board have exact tie overrides; a private royal flush has exact equity one.
`estimate` also computes exact heads-up equity on board quads and board
straights with no possible flush, using range combinations and rank bitmasks.
Board-quads equity against unrestricted multiple opponents uses an exact
combinatorial formula that counts which opponents receive equal kickers.
`estimate_features` lacks the original card identities and therefore skips
these latter shortcuts. Other rare boards can have larger
model errors than aggregate validation metrics suggest. Raw equity uncertainty
does not include incorrect opponent range assumptions.

`evaluate` in `evaluator.py` is an original exact five-to-seven-card bitmask
evaluator. Its integer output compares higher for better hands. The separate
`evaluate_reference` enumerates five-card subsets and uses rank sorting/counts.
The native generator evaluator was checked against 12,000 random Python
evaluations and its exhaustive 2,598,960 five-card category histogram. Python
tests separately compare thousands of hands against the independent reference,
including wheels, two trips, three pairs, seven-card flushes, and ties.

The feature encoder is versioned (`v2-pair-suit-canonical`). Exports with a
different or missing feature version are disabled and `MODEL_STATUS` reports
`feature-version-mismatch`; they are never silently used with another schema.
The module exposes `MODEL_AVAILABLE`, `PREFLOP_AVAILABLE`,
`PREMIUM_AVAILABLE`, and `COMPILED_AVAILABLE`; missing trained
assets activate a rough checkpoint fallback. A finished package must contain
all three JSON assets and the generated compiled module, and verify these
flags. That fallback is not benchmarked
as the trained policy.
