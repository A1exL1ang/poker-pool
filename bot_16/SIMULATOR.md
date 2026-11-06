# Local simulation assumptions and evidence

This engine and opponent suite were authored independently for bot_16. They are
not the official contest simulator. `CONTRACT.md` defines the observation and
action semantics. The public contest rules supply only the initial 10/20 blinds,
2,000-chip starting stack, eight seats, 25-hand blind intervals and 200-hand cap.

- Default later blinds are a clearly labelled **synthetic** progression in
  `simulator.SYNTHETIC_BLINDS`. Pass `blind_schedule` to change it.
- Benchmark point vectors `winner`, `top3` and `linear` are **synthetic sensitivity
  scenarios**, not official announced points. The simulator requires explicit
  payouts to compute point results. At the hand cap it computes exact subset-DP
  ICM over survivors; eliminated players have already received their places.
- Three-or-more-player tables use a forward-moving button, an online convention
  also documented by [PokerStars](https://www.pokerstars.com/help/articles/fwd-moving-button/).
  The heads-up transition adjusts the button to prevent consecutive big blinds.
  TDA live tournaments instead use a dead button. The contest convention is
  unknown and an official engine must supersede this local choice.
- All-in raises below a full increment do not individually reopen action.
  Cumulative short raises reopen separately for a player facing a full increment
  since their last action. A short opening all-in requires the next raiser to
  add the full minimum bet; earlier checkers cannot raise until facing a full bet.
  See [TDA rules](https://www.pokertda.com/view-poker-tda-rules/) and its
  [short-opening discussion](https://www.pokertda.com/forum/index.php?topic=663.0).
- Uncalled contributions are refunded. Each side pot is awarded separately.
  Tied pots assign odd chips clockwise from the button. Simultaneous eliminations
  order by starting stack and split equal-stack finishing points, a local
  convention that the supplied contest text does not specify.
- Burns are omitted: for a uniformly shuffled unseen deck the observed deal
  distribution is equivalent. Agents only receive their own hole cards and the
  current public board/actions; future cards and other hands are never exposed.
- Decision timings include agent policy calls. Enforcement is opt-in during
  strategic experiments and must be reported. Illegal actions/errors fall back
  to check/fold. Internal candidate error/fallback counters are also reported.

Independent test command:

```powershell
python -m unittest discover -s tests -p test_simulator.py -v
python benchmarks/fuzz.py --hands 10000 --seed 160701 --output benchmarks/mechanics_fuzz.json
```

It checks evaluator agreement against a separately written five-card combination
oracle, multiway and heads-up action order, short and cumulative all-ins,
reopening, side pots, refunds, tied-board/odd-chip awards, chip conservation,
observation isolation, timeout/error fallbacks and ICM. A 10,000-hand mechanics
stress run is recorded in `benchmarks/mechanics_fuzz.json`; its result is not a
strategy claim. `benchmarks/prototype_*.json` use development fallback equity
before trained assets and are not final held-out results.

Baselines cover tight aggressive, loose aggressive, nit, calling station,
maniac, random, bounded Monte Carlo equity, all-jam and all-fold policies.
They are heuristic stress opponents and do not approximate unknown contest bots.
The Monte Carlo baseline may exceed 1ms, so its admissibility is not claimed.
All final latency measurements should be isolated from parallel CPU-heavy runs.

```powershell
python benchmarks/tournament.py --tables 200 --seed 160000 --field mixed --output benchmarks/development.json
python benchmarks/tournament.py --tables 200 --seed 160000 --field mixed --candidate baseline:tag --output benchmarks/development_control.json
python benchmarks/tournament.py --tables 200 --seed 160000 --field mixed --params '{"call_margin":0.04}' --output benchmarks/development_variant.json
python benchmarks/tune.py --tables 96 --seed 160000 --output benchmarks/tuning.json
```

The tuning runner shares seeds/deals across variants, stores paired differences,
and rejects changing source/assets during an experiment. Tuning results require
fresh held-out evaluation after freezing policy and assets; the many candidate
comparisons are not independently significant confidence claims.


Additional development stress fields are `adversarial`, `calling`, `minraise`,
`polar`, and `selfplay`. The first four lazily load `baselines_stress.py`;
selfplay uses eight copies of the candidate with independent reset seeds and
rotates the reported seat. These are testing conditions, not other teams' bots.

```powershell
python benchmarks/fuzz.py --stress --hands 10000 --seed 161100 --output benchmarks/stress_fuzz.json
python benchmarks/tournament.py --field adversarial --tables 200 --seed 161100 --output benchmarks/adversarial_development.json
python benchmarks/tournament.py --field selfplay --tables 400 --seed 162100 --output benchmarks/selfplay_development.json
python benchmarks/tournament.py --tables 20 --seed 163100 --observations benchmarks/observations.json --output benchmarks/observation_collection.json
python benchmarks/latency.py --observations benchmarks/observations.json --output benchmarks/latency.json
python benchmarks/package_check.py dist/bot_16.zip --output benchmarks/package_check.json
```

Latency reports measure complete `Agent.act`, including history processing,
features, inference, ICM and action normalization. They include cold first act,
wall/process/thread CPU distributions, raw overrun counts and internal fallback
counts. Legal long-history tests produce 397–421 entries using alternating full
minimum raises at the initial 100BB depth. No timing sample is discarded.
Windows process/thread CPU clocks may tick only every 15.625ms despite reporting
a finer nominal resolution; the harness reports zeros and observed minimum ticks.
Use Linux for meaningful per-decision CPU quantiles, and avoid concurrent compute
for final measurements. Wall scheduling interruptions still count as overruns.

The package check requires all three assets and the matching compiled inference
module, enforces archive and uncompressed sizes below decimal 100MB, rejects
unsafe archive paths and control metadata, and validates the extracted runtime
under `python -I -S` with only extraction plus standard-library import paths.
Socket APIs and socket audit events are denied before importing the bot; this is
a Python-level offline validation guard, not a claim of kernel sandboxing.
