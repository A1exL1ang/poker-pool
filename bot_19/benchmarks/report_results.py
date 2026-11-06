"""Render measured evidence into the deliverable; never invent missing results."""
import json
from pathlib import Path
import re


def read(name):
    return json.loads(Path('measurements',name).read_text())


def main():
    final=read('final_tournaments.json')
    tune=read('tuning.json')
    latency=read('final_latency.json')
    stress=read('final_stress.json')
    enforced=read('final_enforced.json')
    equity=read('equity_final_heldout.json')
    native=read('native_final_validation.json')
    generated=read('equity_codegen_final_linux.json')
    tests=Path('.control/final-tests.log').read_text()
    match=re.search(r'Ran (\d+) tests',tests)
    if not match or not re.search(r'^OK\s*$',tests,re.M):
        raise RuntimeError('final unittest success evidence missing')
    lines=['# Measured results','',
        'These are independent local experiments, not contest scores. The official interface,',
        'payouts and later blind schedule were unavailable. All distributions below are',
        'explicit hypothetical sensitivity cases. No submission, push or publication occurred.','',
        '## Locked held-out tournament evaluation','',
        f"Selected parameters were frozen after {tune['arguments']['candidates']} candidates on tune seeds and independent validation;",
        f"validation paired gain was {tune['validation_paired_gain']:.5f} normalized points (SE {tune['validation_paired_se']:.5f}).",
        'Final evaluation never selects or modifies the policy. It uses fresh seeds, six',
        'opponent pools, five payout cases and three blind schedules. The diverse opponent',
        'pool, top-two/satellite payouts and slow schedule were absent from policy tuning.',
        'All policies and models are original local implementations. The same final model',
        'assets are shared by the applicable reference opponents.','',
        '| Policy | Tables | Mean normalized points | 95% normal CI |',
        '|---|---:|---:|---|']
    for label,c in final['candidates'].items():
        s=c['summary'];ci=s['score_95ci_normal']
        lines.append(f"| {label} | {s['tables']:,} | {s['score_mean']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    pair=final['paired_selected_minus_reference']
    ci=pair['paired_95ci_normal']
    lines+=['',f"Selected minus reference on common deck seeds: **{pair['mean_gain']:+.5f}**",
            f"(paired 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}]).",
            ('The interval supports an improvement under this test distribution.' if ci[0]>0 else
             'The interval does not establish a positive improvement over the reference.'),'',
            'Payout totals are normalized to one; an exchangeable seat averages 0.125.',
            'Averaging across these hypothetical schedules is a local tuning objective.',
            'The intervals describe sampling uncertainty, not generalization to contest bots.','',
            '| Selected policy opponent pool | Tables | Mean points | 95% CI |',
            '|---|---:|---:|---|']
    for name,s in final['candidates']['selected']['by_pool'].items():
        ci=s['score_95ci_normal']
        lines.append(f"| {name} | {s['tables']:,} | {s['score_mean']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    s=final['candidates']['selected']['summary']
    lines+=['',f"Selected policy: {s['decisions']:,} decisions, {s['invalids']} illegal actions,",
            f"{s['errors']} raised exceptions, {s['fallbacks']} internal error fallbacks. The normal",
            f"parallel tournament run recorded {s['overruns']} wall-clock overruns without enforcing them.",'',
            '## Decision timing and fallbacks','',
            f"Final timing platform: `{latency['platform']}`, Python {latency['python']}.",
            'The boundary is in-process `Agent.act`, excluding imports, model loading,',
            'reset, observation construction and JSON/process IO. GC remains enabled.',
            'These finite measurements are not an absolute deadline guarantee.','',
            '| Measurement | Calls | Mean µs | p99 µs | Maximum µs | >1 ms |',
            '|---|---:|---:|---:|---:|---:|']
    for name,key in [('Real gameplay trace','trace_decisions'),('Cold-history replay','replay_wall')]:
        s=latency[key]
        lines.append(f"| {name} | {s['count']:,} | {s['mean_us']:.1f} | {s['p99_us']:.1f} | {s['max_us']:.1f} | {s['over_1ms']} |")
    for name,c in stress['cases'].items():
        s=c['wall']
        lines.append(f"| {name} | {s['count']:,} | {s['mean_us']:.1f} | {s['p99_us']:.1f} | {s['max_us']:.1f} | {s['over_1ms']} |")
    e=enforced['summary']
    lines+=['',f"Separate enforced-deadline run: {e['tables']:,} tables, {e['decisions']:,} hero decisions,",
            f"**{e['overruns']} timeout check/fold replacements**, {e['errors']} exceptions,",
            f"{e['invalids']} illegal actions and {e['fallbacks']} internal fallbacks.",
            f"Mean normalized points in that different-seed run: {e['score_mean']:.5f}.",
            'Timeout-enforced outcomes depend on OS scheduling and are therefore not exactly reproducible.',
            'Earlier concurrent Windows runs also exceeded 1 ms; their original JSON reports',
            'are retained. Windows per-call thread CPU readings were quantized to roughly',
            '15.625 ms and cannot identify the cause of individual stalls.','',
            '## Equity accuracy and mechanics','',
            'The final model was selected using validation only. A fresh 30,000-state set',
            '(seed 1900205, 8,192 simulation draws per applicable state) was then evaluated.',
            'Errors below measure the declared synthetic range distribution, not strategy quality.','',
            '| Street | Test states | MAE | RMSE | 99th-percentile absolute error |',
            '|---|---:|---:|---:|---:|']
    for street,name in [('3','Flop'),('4','Turn'),('5','River')]:
        r=equity['streets'][street]
        lines.append(f"| {name} | {r['rows']:,} | {r['mae']:.5f} | {r['rmse']:.5f} | {r['p99_absolute_error']:.5f} |")
    lines+=['',
            f"- {match.group(1)} final unit tests passed, including independent side-pot/ICM checks,",
            '  minimum raises/reopening, short all-in blinds, heads-up/dead-button rules,',
            '  state reset, card secrecy, public observations and offline decision calls.',
            '- Exhaustive 2,598,960 five-card rank census plus independent native/Python',
            '  evaluator and feature comparisons passed.',
            f"- Generated inference: {generated['actual_hands']+generated['random_feature_vectors']+generated['float32_boundary_cases']:,}",
            f"  comparisons, maximum difference {generated['max_fast_reference_difference']} against the reference.",
            '- A separately calibrated multiway correction reduced isolated test RMSE',
            '  from 0.05483 to 0.03501. Board-playing tie handling has separate tests.',
            '- Large individual equity errors remain; see maximum and tail errors in JSON.',
            '','## Reproduction and limits','',
            'See `training/final_job.sh` for the exact final command sequence; full per-table',
            'rows, seeds, source/model hashes and sensitivity breakdowns are in `measurements/`.',
            '`training/tune_job.sh`, `training/refine_job.sh` and `training/job.sh` preserve',
            'the training/selection commands. `build_package.py` checks model assets, creates',
            'a deterministic offline ZIP and smoke-tests its fresh extraction using isolated',
            'Python with networking blocked and decision-time file access blocked.',
            '','The strategy remains a compact range model and heuristic action policy, with',
            'approximated opponent behavior and future realization. It has no equilibrium,',
            'exploitability, official compatibility, contest-win or absolute timing guarantee.']
    Path('RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
