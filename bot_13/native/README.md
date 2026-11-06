This directory contains original C++17 code for bot13. There are no third-party
poker engines, learned policies, or lookup tables in the source. It implements
standard five-to-seven-card Hold'em hand ordering, fixed-rollout range equity,
board-relative range features, and exact one-player independent-chip-model equity.

Build ahead of gameplay from the workspace root:

```
python native/build.py
python -m unittest discover -s tests -p test_native.py -v
python native/benchmark.py --iterations 1000 --output native/benchmark_local.json
```

The build uses g++ or clang++ with C++17 and -O3. Windows produces bot13.dll with
statically linked compiler runtimes. Linux produces libbot13.so with -fPIC and
hidden internal symbols. No compiler, subprocess, file writes, or network access
are used when loading the prebuilt library or making decisions. The Python
wrapper reports unavailable native support explicitly; it does not silently run
an unbounded Python Monte Carlo calculation.

Cards use `4*(rank-2)+suit`, with rank 2..14 and suits c,d,h,s encoded 0..3.
`b13_evaluate` accepts 5..7 distinct cards and returns zero on invalid input.
Otherwise a larger score wins. Category is `score >> 24` (0 high card through 8
straight flush), followed by conventional rank values in descending tiebreak
nibbles at bits 20,16,12,8,4. This is a local ABI, not an official contest interface.

`NativeEvaluator.equity(hero, board, opponents, weights, seed, samples)` returns
showdown pot share and its Monte Carlo standard error. Weights, when supplied,
are one flat array of opponents*1326 doubles. Each 1326-array is lexicographic
card combinations `i<j`, index `i*(103-i)//2+j-i-1`. Hero and board blockers are
removed before range normalization. Each opponent is independently drawn from
its range, then the entire tuple is rejected on any shared card. This preserves
the product-of-ranges law conditional on disjoint cards. At most 32 tuple attempts
are made per requested rollout. The native function returns the accepted count;
the wrapper exposes it as last_accepted. Extremely concentrated incompatible
ranges may yield no accepted rollout and raise ValueError, which the caller must
handle. Every accepted rollout draws a uniformly random legal board completion.
Split pots receive the correct reciprocal share. HU river equity enumerates all
legal weighted opposing hands exactly, regardless of requested rollout count;
its reported Monte Carlo error is zero.

`modeled_equity(hero, board, specs, seed, samples)` builds ranges in native code.
Each spec is `(preflop_top_fraction, postflop_strength_threshold, slope,
bluff_floor, draw_weight)`. It combines a smoothed learned preflop percentile
range with a board-relative hand-strength percentile and simple private-card
flush/straight draws. This inference is a heuristic, not a solver or calibrated
opponent posterior. Off-range mass acknowledges uncertainty. Current strength
is an exact uniform-opponent percentile on the visible board, not future equity.
Returned prefix shares use the same full joint opponent samples and report
equity versus the first k supplied opponents; callers can sort by contributions
and use these for side-pot layers. These prefix marginals remain conditioned on
all modeled opponents' jointly disjoint hands, including blockers.

Train our own 169-class preflop equity asset offline:

```
python tools/train_preflop.py --samples 200000 --seed 130613 --output assets/preflop.json
```

Pass one computes HU uniform-opponent equities and ranks all 1326 combinations by
that equity. Pass two computes uniform multiway equities for 2..8 total players
and HU equities against the top 10%,20%,35%,50%,100% of combinations. Complete
tied hand classes are included at the range boundary, avoiding suit bias. The
JSON records its seed, rollout count and range fractions; `hands[key].random`
has seven entries versus 1..7 random opponents and `hands[key].ranges` follows
range_fractions. Load `set_preflop_strength(table['hands'])` once before decisions
to give modeled native ranges the same learned ordering. Without initialization,
a documented simple rank/pair/suited/gap approximation provides a fallback order.

`icm_equity(stacks, payouts, hero)` computes exact ICM for caller-provided payouts
and 1..8 players using subset dynamic programming. Zero-stack players divide
unresolved trailing rank equity uniformly. Known eliminated finishing positions
must be assigned by the caller before removing them from a surviving-player ICM.
No contest payout structure or blind schedule is assumed by this native layer.

Benchmark files record local measurements, including operating-system scheduling
overruns. Fixed bounded computation does not promise hard real-time execution.
