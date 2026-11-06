# Experiment ledger

This directory holds local evidence, not contest results. Official interface,
payouts and later blinds were unavailable. All payout and blind schedules in
these reports are explicitly synthetic.

## Development and tuning

- `prototype_smoke.json`: first runnable fallback prototype, before real assets.
- `v1_pilot_mixed.json`: 40-table development pilot; small and not held out.
- `equity_validation_v1.json`: 300,000 training states, seed 140120; external
  validation 24,000 states, seed 914007, with 2,048 trials or exact HU river labels.
  Overall mean absolute error 0.0316067.
- `equity_validation_v2.json`: 1,200,000 training states, seed 140220, 20%
  structural oversampling, 58 features, 300 trees/63 leaves per street. Same
  external development validation. MAE 0.0233421, RMSE 0.0378194. External labels
  were not used in model fitting; choosing v2 after viewing them makes this
  development/selection evidence.
- The v1/v2 paired development tournaments use seeds 141000 through 141499
  in each named field. Configuration tuning uses a further explicitly separate
  development range beginning at 142000. Reports retain table-level rows.
- `tuning.json`: 44,000 tables, 22 configurations across four fields.
- `confirmation.json`: 35,000 additional development tables, five configurations
  and ten opponent/scoring/blind cases; seed family beginning at 144000.
- `selection.json`: the selected `wider_short_no_nash` policy and the explicit
  reason for choosing faster inference over a statistically indistinguishable
  rollout alternative. Its aggregate paired development gain was 0.00590461
  points per table (SE 0.00284506). These intervals are unadjusted selection
  evidence. They are not proof of held-out improvement.

## Timing interpretation

Timing reports record every wall overrun, including scheduling delays. Import,
asset loading, model initialization, observation construction and serialization
are excluded from timed `Agent.decide` calls. Fresh per-observation state avoids
benefit from repeated feature caches. This is a local interpretation of the
1 ms rule and needs reconciliation with the official timing harness.

The Windows process CPU clock quantizes short calls too coarsely for per-call
quantiles; reports set `cpu_quantiles_usable=false`. Wall measurements remain
reported, including runs with overruns. Linux reports include usable process CPU
measurements and platform details. The internal budget fallback checks/calls
after a priced continuation decision, whereas an engine-enforced timeout checks
or folds. Their counts are distinct.

## Final evaluation

The selected gameplay source and assets were frozen in
[`locked_candidate.json`](locked_candidate.json) before untouched data was
generated. Collection after the interrupted CLI verified every runtime file
against that lock. No gameplay policy or asset changed during evaluation or
resumed collection. [`final_summary.json`](final_summary.json) collects the
verified results and limitations.

All **105 tests passed** remotely. The final selected-policy evaluation covers
**17,500 tables and 16,888,024 decisions across all seats**, with zero illegal
actions and zero exceptions. Another 8,000 paired reference tables cover
7,879,027 decisions, also with zero illegal actions or exceptions.

[`heldout_tournaments.json`](heldout_tournaments.json) contains full table rows,
seed ranges, strict timing enforcement, payout/blind sensitivity cases, and
paired reference comparisons. Each primary field has 2,000 tables with the
synthetic top-three payout and moderate blind schedule. Mean normalized points
and approximate 95% normal intervals are:

- Mixed: **0.35244** [0.34395, 0.36093].
- Rollout: **0.31154** [0.30280, 0.32027].
- Aggressive: **0.35477** [0.34710, 0.36245].
- Tight: **0.33786** [0.32853, 0.34718].
- Loose: **0.36346** [0.35617, 0.37076].
- Self field against seven copies of the fixed reference: **0.13010**
  [0.12213, 0.13807].

The equal-share reference is 0.125, since synthetic payouts total one. These
opponents are original local test policies, not submitted contest bots. The
selected policy's paired gain over the fixed reference in the mixed field was
0.01739 [0.01005, 0.02472]. The self-field paired difference was 0.00195
[-0.00706, 0.01096], which does not establish an advantage there.

[`equity_heldout.json`](equity_heldout.json) evaluates the unchanged v2 forest on
30,000 untouched random states, seed 2014014, with 4,096 rollout trials or exact
HU river labels. MAE is **0.0230217**, RMSE **0.0375831**. Labels still depend on
the explicitly synthetic opponent range model. Exact runtime HU river
enumeration bypasses the forest in those positions.

[`pushfold_heldout.json`](pushfold_heldout.json) audits the optional constrained
heads-up policy against a separate 20,000-trial-per-pair matrix, seed 28914014.
Maximum estimated finite-game NashConv is **0.0003484 BB/hand**, compared with
the much smaller training-matrix gap. Matrix sampling uncertainty remains, and
neither number measures full NLHE exploitability. This option is disabled in
the selected defaults.

## Final timing and fallbacks

[`latency_final_linux.json`](latency_final_linux.json) measures **96,560** calls
on a fresh, balanced corpus of 4,828 observations, including short-stack/full-rank
ICM stress. Native wall mean was **98.9 us**, p99 **168.2 us**, p99.9 **296.3 us**.
There were **2 calls above 1 ms**, maximum **4,184.0 us**, and **2 internal budget
fallbacks**. CPU timing also exceeded 1 ms on those two calls; their cause was
not instrumented, so they are not attributed solely to scheduling. Native equity
backend fallbacks were zero.

[`latency_portable_linux.json`](latency_portable_linux.json) forces the Python
path with `native=false` on the same corpus: **24,140** calls, p99 **518.9 us**,
**5 calls above 1 ms**, maximum **4,789.6 us**, and **5 internal budget fallbacks**.
A separate isolated archive check also verified operation with native binaries
absent. The replay corpus is preserved as
[`latency_final_corpus.jsonl.gz`](latency_final_corpus.jsonl.gz).

The strict held-out tournaments recorded **9,810 engine timeout fallbacks across
all seats** in the selected-policy tables, plus 6,328 in the paired reference
tables. Per-seat timeout attribution was not recorded, so these are not claimed
as the selected bot's individual overrun counts. The selected Agent recorded
1,205 internal budget fallbacks in its 17,500 tables. These figures concern a
different, concurrent live-play workload from isolated replay timing.

The native implementation has ample typical timing margin, but **universal
1 ms compliance is not claimed**. Earlier Windows and development reports with
overruns remain available. Contest hardware, startup treatment, serialization,
and the official timing harness remain untested.

## Offline package

[`package_validation.json`](package_validation.json) records successful isolated
imports and decisions with file/network/process access blocked after startup,
both with the native backend and with native binaries omitted. Runtime content
matches the candidate lock and the archive's SHA-256 and CRC checks passed after
collection.

- Archive: [`../bot_14_runtime.zip`](../bot_14_runtime.zip), **1,396,052 bytes**.
- Runtime payload: **4,041,281 bytes**, 16 files plus the embedded manifest.
- SHA-256: `30c9b619c124c1bd35f592cc39f2ec49b39004b8a2f4077d6c8d9f084ce83886`.
- File manifest: [`../bot_14_manifest.json`](../bot_14_manifest.json).

No training, replay, or evaluation was restarted during resumed collection.
The frozen policy is ready for local collection. Official interface integration
is still required; no contest submission, push, or publication was performed.
