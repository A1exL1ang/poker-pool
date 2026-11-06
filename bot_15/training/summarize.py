"""Create a factual final report from retained measurements; no new selection."""
import json
from pathlib import Path


def read(name):
    return json.loads(Path('artifacts', name).read_text())


def interval(metric, scale=1):
    lo, hi = metric['ci95_normal']
    return f"{metric['mean'] * scale:.2f} [{lo * scale:.2f}, {hi * scale:.2f}]"


def main():
    policy = read('policy.json')
    held = read('heldout.json')
    enforced = read('heldout_enforced_subset.json')
    linux = read('latency_linux_final.json')
    windows = read('latency_windows_final.json')
    fuzz = read('integration_fuzz_final.json')
    equity = read('equity_runtime_metrics.json')
    tuning = read('policy_tuning.json')
    refinement = read('policy_refinement.json')
    lines = [
        '# Final measurements — bot_15', '',
        'This is local experimentation, not contest submission or proof of official interface compatibility. The policy was frozen before the held-out scenarios below. No held-out result was used to change it.', '',
        '## Frozen candidate', '',
        f"Selected `{policy['candidate']}` from tuning only: `{json.dumps(policy['config'], sort_keys=True)}`.", '',
        f"Initial sweep: {len(tuning['results']) * tuning['tables_per_cell']:,} table runs; confirmation: {len(refinement['results']) * refinement['tables_per_cell']:,}. The predeclared selection objective was 80% average scenario score and 20% weakest-population average, normalized by equal-share points. All opponent pools and payout examples are synthetic. Tuning rotated absolute seats in lockstep with the starting button, so the initial relative position was fixed. This limitation was found before final reporting; final evaluation crosses all64seat/button pairs every64tables. The first incomplete evaluation was discarded, and the policy was not changed.", '',
        f"The selected policy's paired confirmation gain over the untuned policy was {interval(policy['paired_tuning_difference_vs_untuned'])} normalized units (95% normal interval). That interval includes zero: tuning does not establish a reliable improvement over the untuned policy.", '',
        '## Held-out strategy evaluation', '',
        f"{held['scenarios']} scenarios, {held['tables_per_scenario']:,} tables per scenario per policy, three policies: {held['scenarios'] * held['tables_per_scenario'] * 3:,} table runs. Candidate seats rotate. Opponents, per-hand decks, and table seeds are paired across policies. Eliminations can cause later paths to diverge. Intervals are descriptive 95% normal intervals, without multiple-comparison correction.", '',
        'All point vectors below are local sensitivity fixtures, scaled to total 100: winner-take-all; top3=50/30/20; flat3=34/33/33; linear=8:7:6:5:4:3:2:1 normalized to 100. Equal share is 12.5. Default blinds double every 25 hands; gentle blinds use 10/20,15/30,25/50,40/80,60/120,100/200,150/300,250/500. Neither full progression is claimed official.', '',
        '| Field / points / blinds | Candidate points [95% CI] | Difference vs balanced baseline [95% CI] | Difference vs untuned bot [95% CI] |',
        '|---|---:|---:|---:|',
    ]
    summaries = []
    for r in held['results']:
        kind, payout, schedule, _ = r['scenario']
        m = r['policies']['candidate']['metrics']['points']
        balanced = r['paired_differences']['balanced']['points']
        untuned = r['paired_differences']['untuned']['points']
        lines.append(f"| {kind} / {payout} / {schedule} | {interval(m,100)} | {interval(balanced,100)} | {interval(untuned,100)} |")
        summaries.append(dict(scenario=r['scenario'], candidate=m, vs_balanced=balanced, vs_untuned=untuned))
    lines += ['',
        '`challenge` consists of independently written equity-aware and polarized betting policies that were not used for tuning. They share the equity asset, so estimation errors are correlated. `mirror` uses seven copies of the untuned bot. Both are heuristics, not professional or equilibrium opponents. No average here predicts contest ranking.', '',
        '## Full-decision latency and fallback behavior', '',
        'Timing includes the complete in-process `Agent.act()` path with garbage collection enabled. Imports, asset loading/validation, JSON parsing, and I/O are excluded. Replay observations come from 200 separate tables; repeated replays are not independent strategic samples. OS scheduling and garbage collection remain part of wall time.', '',
        '| Machine | Calls | Median us | p99 us | p99.9 us | Maximum us | Calls >1 ms | Startup ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in [('Linux compute node / Python ' + linux['python'], linux), ('Windows desktop / Python ' + windows['python'], windows)]:
        w = r['wall']
        lines.append(f"| {name} | {w['n']:,} | {w['p50_us']:.1f} | {w['p99_us']:.1f} | {w['p999_us']:.1f} | {w['max_us']:.1f} | {w['over_1ms']} | {r['constructor_us']/1000:.1f} |")
    lines += ['',
        f"Internal error fallbacks during latency replays: Linux {linux['internal_error_fallbacks']}, Windows {windows['internal_error_fallbacks']}. First decision after reset: Linux max {linux['first_decision']['max_us']:.1f} us, Windows max {windows['first_decision']['max_us']:.1f} us. The Windows per-call CPU clock is too coarsely quantized for useful quantiles; wall time is the relevant reported measurement.", '',
        '**The 1 ms bound is not guaranteed:** measured overruns are retained above, not discarded. The local engine implements the rule fallback (check if free, otherwise fold) for exceptions or over-budget calls. A collector must rerun timing with the actual adapter, hardware, and initialization policy.', '']
    counts = {k: 0 for k in ('decisions', 'over_1ms', 'internal_fallbacks')}
    errors = invalid = 0
    for r in enforced['results']:
        candidate = r['policies']['candidate']
        for k, v in candidate['candidate_counts'].items():
            counts[k] += v
        errors += sum(x['table_errors'] for x in candidate['rows'])
        invalid += sum(x['table_invalid'] for x in candidate['rows'])
    lines += [f"Separate 1 ms-enforced stress evaluation: {enforced['scenarios'] * enforced['tables_per_scenario']:,} candidate tables, {counts['decisions']:,} candidate decisions, {counts['over_1ms']:,} timed candidate fallbacks, {counts['internal_fallbacks']} internal candidate error fallbacks, {errors} table engine errors and {invalid} illegal actions. This uses a subset of held-out seeds and eight concurrent workers; it is not an independent extra strategy sample. Complete results are in `artifacts/heldout_enforced_subset.json`.", '',
        '## Mechanics, model and offline packaging', '',
        '- 69 unit/regression tests passed: betting order, heads-up/dead-button rules, full and short raises/reopening, pot eligibility, ties/odd chips, simultaneous eliminations, point/chip conservation, exact ICM reference comparisons, forced chops, blind-only all-ins, table reset, suit/seat relabeling and balanced benchmark seating.',
        '- Exhaustive independent evaluator audit: all 2,598,960 five-card hands plus 10,000 seven-card subset comparisons; 19,664 native/Python feature comparisons including every quads turn board with all legal hole pairs. Numeric counts and scripts are retained.',
        f"- Final integration fuzz: {fuzz['config']['tables']:,} tables, {fuzz['totals']['table_decisions']:,} total actions, {fuzz['totals']['candidate_decisions']:,} candidate actions, {fuzz['totals']['invalid_actions']} illegal actions, {fuzz['totals']['engine_errors']} engine errors, {fuzz['totals']['candidate_internal_fallbacks']} internal candidate fallbacks. Unenforced wall overruns: {fuzz['totals']['candidate_wall_clock_over_1ms']}.",
        f"- Equity model: 240,000 training hands, 12,000 validation, 12,000 held-out. The eight-tree runtime was selected on validation accuracy/latency. Held-out MAE {equity['mae']:.6f}, RMSE {equity['rmse']:.6f}, 95th percentile absolute error {equity['p95_absolute_error']:.6f}. Labels contain Monte Carlo noise. Asset size 5,160,408 bytes.",
        '- The gameplay ZIP contains only whitelisted stdlib Python sources, numeric assets and documentation. The packaging command verifies hashes, extracts into a fresh directory, and runs import/action/reset checks under isolated Python with all socket audit events forbidden. See `artifacts/package_verification.json` for actual ZIP size and SHA-256.', '',
        '## Limits and stopping rationale', '',
        'No official starter interface, complete blind schedule or payout structure was available. No official compatibility or payout claims are made. The bot uses approximate opponent ranges, pot-layer multiway equity, draw/realization models and binary ICM risk scenarios. It does not solve the full game, account exactly for correlated multiway holdings or all future betting, or guarantee safety against adversarial strategies. Observation histories can omit actions after the bot folds, limiting opponent estimates. State must be reset at every table.', '',
        'The larger 24-tree model improved validation MAE by only 0.00085 while substantially increasing latency. The confirmation sweep found modest and statistically uncertain gains among nearby policies. With independent mechanics checks, held-out stress scenarios, offline packaging and timing measurements complete, further local parameter search has diminishing evidentiary value without the actual contest interface and opponent population. No absolute optimality claim is made.', '',
        '## Reproduce', '',
        '```text',
        'python -m unittest discover -s tests -v',
        'python -m training.tune_policy --tables 128 --workers 8',
        'python -m training.tune_policy --stage refinement --tables 512 --workers 8 --output artifacts/policy_refinement.json',
        'python -m training.freeze_policy',
        'python -m benchmarks.heldout --tables 1000 --workers 8',
        'python -m benchmarks.heldout --tables 200 --workers 8 --enforce-1ms --output artifacts/heldout_enforced_subset.json',
        'python -m benchmarks.fuzz --tables 1000 --seed 7410001 --output artifacts/integration_fuzz_final.json',
        'python -m benchmarks.latency --tables 200 --repeat 5 --output artifacts/latency_local.json',
        'python -m training.package_bot',
        '```', '',
        'All raw table rows, source hashes, tuning results and timing records remain in `artifacts/`. Remote work used only newly requested bounded bot_15 jobs in its authorized private directory; resource bookkeeping is in `.control/`. Nothing was pushed, published or submitted.', '']
    Path('MEASUREMENTS.md').write_text('\n'.join(lines))
    Path('artifacts/final_summary.json').write_text(json.dumps(dict(
        config=policy['config'], heldout=summaries, linux_latency=linux['wall'],
        windows_latency=windows['wall'], enforced_candidate_counts=counts,
        integration=fuzz['totals'], equity={k: equity[k] for k in ('n','mae','rmse','p95_absolute_error')}), indent=2))
    print('Wrote MEASUREMENTS.md and artifacts/final_summary.json')


if __name__ == '__main__':
    main()
