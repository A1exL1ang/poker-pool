"""Run the frozen held-out suite on our bounded remote allocation only.

This intentionally performs substantial computation. Do not run on the laptop
while the user's quiet-work restriction is active. Policy selection must be
finished before using these disjoint seeds or the held-out equity test file.
"""
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.tournament import run, fingerprints


def task(spec):
    label, options = spec
    path = ROOT / 'benchmarks' / ('heldout_' + label + '.json')
    if path.exists():
        previous = json.loads(path.read_text())
        same = (previous.get('file_sha256') == fingerprints()
                and previous.get('tables') == options['tables']
                and previous.get('seed_start') == options['seed']
                and previous.get('candidate') == options.get('candidate', 'agent:Agent')
                and previous.get('field') == options['field']
                and previous.get('synthetic_payout_label') == options.get('payout', 'top3')
                and previous.get('enforce_time_limit') == options.get('enforce_time_limit', False)
                and not previous.get('files_changed_during_run')
                and not previous.get('candidate_internal_errors'))
        if same:
            return label, previous
    report = run(**options)
    path.write_text(json.dumps(report, indent=2))
    if report['files_changed_during_run'] or report['candidate_internal_errors']:
        raise RuntimeError('Unfrozen source or candidate error: ' + label)
    return label, report


def main():
    frozen = fingerprints()
    freeze_path = ROOT / 'benchmarks/freeze.json'
    if freeze_path.exists():
        freeze = json.loads(freeze_path.read_text())
        if freeze['source_sha256'] != frozen:
            raise RuntimeError('Frozen sources changed: do not reuse held-out seeds for tuning')
    else:
        freeze = {'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'source_sha256': frozen, 'purpose': 'Held-out evaluation after model/policy selection'}
        freeze_path.write_text(json.dumps(freeze, indent=2))
    # Threaded numerical libraries are not used by gameplay. Four independent
    # single-process simulations stay inside this job's four allocated CPUs.
    specs = []
    for i, (field, tables) in enumerate([
        ('mixed', 1000), ('tag', 1000), ('aggressive', 1000), ('tight', 500),
        ('adversarial', 500), ('calling', 500), ('polar', 500), ('selfplay', 400), ('jam', 300)]):
        options = dict(tables=tables, seed=900000 + i * 10000, field=field)
        if field == 'mixed':
            options['observation_path'] = str(ROOT / 'benchmarks/observations_final.json')
        specs.append((field, options))
    specs.extend([
        ('winner', dict(tables=500, seed=1000000, field='mixed', payout='winner')),
        ('linear', dict(tables=500, seed=1010000, field='mixed', payout='linear')),
        ('doubling', dict(tables=500, seed=1020000, field='mixed',
                          blind_schedule=[(10 * 2 ** i, 20 * 2 ** i) for i in range(8)])),
        ('control_mixed', dict(tables=300, seed=900000, field='mixed', candidate='baseline:tag')),
        ('control_tag', dict(tables=300, seed=910000, field='tag', candidate='baseline:tag')),
    ])
    reports = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        for label, report in pool.map(task, specs):
            reports[label] = report
            print(label, report['points'], 'errors', report['candidate_internal_errors'], flush=True)
    # Everything below runs serially after all simulation workers have joined.
    # Strict deadline tournament uses TAG opponents (cheap independent policies)
    # so slower Monte Carlo reference opponents do not distort that comparison.
    label, report = task(('deadline_tag', dict(tables=600, seed=1030000, field='tag', enforce_time_limit=True)))
    reports[label] = report
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                   check=True, stdout=open(ROOT / 'benchmarks/final_unit_tests.txt', 'w'), stderr=subprocess.STDOUT)
    subprocess.run([sys.executable, 'training/evaluate.py', '--test', 'training/test.csv',
                    '--output', 'benchmarks/equity_heldout.json', '--latency-repeats', '1'], check=True)
    subprocess.run([sys.executable, 'benchmarks/latency.py', '--observations', 'benchmarks/observations_final.json',
                    '--repeats', '5', '--random-hands', '256', '--stress-repeats', '1000',
                    '--output', 'benchmarks/latency_final_linux.json'], check=True)
    if fingerprints() != frozen:
        raise RuntimeError('Source changed after freeze')
    aggregate = {'freeze': freeze, 'reports': {k: {f: r[f] for f in (
        'tables', 'seed_start', 'seed_end', 'synthetic_payout_label', 'points', 'equal_share_points',
        'candidate_violation_count', 'candidate_decisions', 'candidate_overruns_1ms',
        'candidate_internal_errors', 'candidate_internal_fallbacks', 'latency_us', 'elapsed_seconds')}
        for k, r in reports.items()},
        'note': 'Synthetic opponents, later blinds and payouts. Parallel strategy timing is contended; isolated timing is in latency_final_linux.json.'}
    (ROOT / 'benchmarks/heldout_summary.json').write_text(json.dumps(aggregate, indent=2))
    eq = json.loads((ROOT / 'benchmarks/equity_heldout.json').read_text())
    lat = json.loads((ROOT / 'benchmarks/latency_final_linux.json').read_text())
    lines = ['# Measured results', '', 'The model and policy were frozen before these held-out seeds were used.',
             'All final substantial tests ran in a new bounded bot_16 Engaging allocation; the laptop remained quiet.',
             '', '## Held-out local tournaments', '',
             'These are independently authored synthetic opponents and synthetic point schedules, not contest results.',
             'Strategy runs were parallel; their latency counts retain scheduling contention. The deadline TAG run was serial.',
             '', '| Field / test | Tables | Mean points | 95% normal CI | Equal share |',
             '|---|---:|---:|---|---:|']
    for name, r in reports.items():
        lo, hi = r['points']['ci95_normal']
        lines.append(f"| {name} | {r['tables']} | {r['points']['mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {r['equal_share_points']:.3f} |")
    lines += ['', '## Equity accuracy', '',
              f"Held-out Monte Carlo target MAE: {eq['overall']['mae']:.6f}; RMSE: {eq['overall']['rmse']:.6f}; p99 absolute error: {eq['overall']['abs_error_p99']:.6f}.",
              'Targets describe explicit synthetic ranges. Remaining model/range errors and sequential multiway sampling are limitations, not eliminated by this test.',
              '', '## Decision timing', '',
              f"Full-act measured calls: {lat['total_decision_calls']}; wall-time calls above 1 ms: {lat['total_wall_overruns_1ms']}.",
              'All observed scheduling outliers are retained. Per-workload wall and CPU distributions, cold first call, import cost, and error counters are in benchmarks/latency_final_linux.json.',
              'Import/model initialization is outside the measured action deadline. The official hardware and timing boundary are unknown; this does not establish official timing compliance.',
              '', '## Validation and limitations', '',
              'Unit test output: benchmarks/final_unit_tests.txt. Offline ZIP validation: benchmarks/package_check.json.',
              'Gameplay uses only the Python standard library and packaged assets; it makes no online calls.',
              'The simulator uses a documented local contract and forward-moving button. No official interface, payouts, or later blind schedule was supplied.',
              'Earlier model and policy tuning used different seeds and is recorded separately in training/ and benchmarks/tuning_v1.json / tuning_v3.json.',
              'This is a tested empirical policy, not an equilibrium solver or a claim of optimal contest performance.', '']
    (ROOT / 'MEASUREMENTS.md').write_text('\n'.join(lines))
    (ROOT / 'dist').mkdir(exist_ok=True)
    files = [ROOT / p for p in ('agent.py', 'run_bot.py', 'README.md', 'CONTRACT.md', 'SIMULATOR.md', 'MEASUREMENTS.md')]
    files += sorted((ROOT / 'bot').glob('*.py'))
    files += [ROOT / 'assets' / p for p in ('equity_model.json', 'preflop.json', 'preflop_narrow.json')]
    with zipfile.ZipFile(ROOT / 'dist/bot_16.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    subprocess.run([sys.executable, 'benchmarks/package_check.py', 'dist/bot_16.zip',
                    '--observations', 'benchmarks/observations_final.json',
                    '--output', 'benchmarks/package_check.json'], check=True)
    print('FINAL SUITE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
