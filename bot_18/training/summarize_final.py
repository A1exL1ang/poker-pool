"""Summarize completed, frozen local experiments without selecting a policy."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def main():
    inputs=('final_standard','final_hidden_payouts','final_fast_blinds')
    reports=[json.loads((ROOT/'measurements'/f'{name}.json').read_text()) for name in inputs]
    assert all(r['complete'] and r['mode']=='evaluate' for r in reports)
    cells=[]
    total={k:0 for k in ('tables','decisions','errors','illegal','candidate_fallbacks','overruns')}
    for name,report in zip(inputs,reports):
        for row in report['results']:
            assert row['timeout_enforced']
            assert not row.get('source_changed_during_run',False)
            for key in total:
                total[key]+=row.get(key,0)
            cells.append({'experiment':name,'suite':row['suite'],'payouts':row['payouts'],
                          'tables':row['tables'],'points':row['points'],
                          'decisions':row['decisions'],'overruns':row['overruns'],
                          'latency_us':row['latency_us'],'internal_fallbacks':row.get('candidate_fallbacks',0)})
    total['timeout_fraction']=total['overruns']/max(1,total['decisions'])
    finalmodel=json.loads((ROOT/'training'/'final_evaluation.json').read_text())
    timing=json.loads((ROOT/'measurements'/'final_latency_linux.json').read_text())
    frozen=json.loads((ROOT/'measurements'/'frozen_candidate.json').read_text())
    assert finalmodel['model_sha256']==frozen['source_sha256']['equity/model.json']
    for report in reports:
        for row in report['results']:
            assert row['source_sha256']==frozen['source_sha256']
    summary={'scope':'LOCAL SYNTHETIC EVALUATION; no official interface/payout compatibility',
             'frozen_candidate':frozen,'totals':total,'cells':cells,
             'model_test':{k:finalmodel[k] for k in ('overall','by_street','by_opponents','seed','label_trials','data_sha256')},
             'latency_replay':timing,
             'limitations':['Synthetic opponents/payouts/blind schedule; no contest ranking prediction.',
                            'Model ranges are preflop abstractions; postflop selection is heuristic.',
                            '1ms wall overruns trigger actual simulated check/fold; none are discarded.',
                            'Startup, serialization and unavailable official adapter are outside act timing.']}
    output=ROOT/'measurements'/'final_summary.json'
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    wall=timing['wall_us']
    lines=['## Final frozen-candidate measurements','',
           f"The frozen candidate completed **{total['tables']:,} held-out local tournaments** "
           f"and **{total['decisions']:,} decisions** with the 1 ms fallback enforced. "
           f"There were {total['errors']} external exceptions, {total['illegal']} illegal actions, "
           f"{total['candidate_fallbacks']} internal error fallbacks and **{total['overruns']} timing overruns** "
           f"({100*total['timeout_fraction']:.5f}%). Overruns became checks/folds in the simulator.",'',
           f"Separate Linux replay/stress timing ({wall['n']:,} calls): median {wall['p50']:.1f} µs, "
           f"p99 {wall['p99']:.1f} µs, p99.9 {wall['p999']:.1f} µs, maximum {wall['max']:.1f} µs; "
           f"{wall['over_1ms']} calls exceeded 1 ms. Initialization and serialization are excluded. "
           "These are measured results, not a hard real-time guarantee.",'',
           f"The independent equity test used {finalmodel['overall']['n']:,} fresh states with "
           f"{finalmodel['label_trials']:,} Monte Carlo trials each: MAE "
           f"{100*finalmodel['overall']['mae']:.2f} percentage points, RMSE "
           f"{100*finalmodel['overall']['rmse']:.2f} points. The 95th-percentile absolute error was "
           f"{100*finalmodel['overall']['p95_absolute']:.2f} points and the largest was "
           f"{100*finalmodel['overall']['max_absolute']:.2f} points, so individual rare-state estimates "
           "can be substantially wrong. These targets use the documented range abstraction.",'',
           'Standard-schedule synthetic 50/30/20/0/0/0/0/0 point results (512 tables per row):','',
           '| Opponent suite | Mean points | Approximate 95% interval |',
           '|---|---:|---:|']
    for r in cells:
        if r['experiment']=='final_standard' and r['payouts']==[50,30,20,0,0,0,0,0]:
            ci=r['points']['normal_95_ci']
            lines.append(f"| {r['suite']} | {r['points']['mean']:.2f} | {ci[0]:.2f}–{ci[1]:.2f} |")
    lines += ['', 'Equal allocation of this synthetic point pool is 12.5 per seat. These opponents are local '
              'heuristics or variants of this bot; the figures do not establish contest rank or optimality. '
              'All payout/blind sensitivity cells and source hashes are in `measurements/final_summary.json`.','']
    readme=ROOT/'README.md'
    text=readme.read_text(encoding='utf-8')
    marker='## Final frozen-candidate measurements'
    if marker in text:
        text=text[:text.index(marker)]
    readme.write_text(text.rstrip()+'\n\n'+'\n'.join(lines),encoding='utf-8')
    print(json.dumps({'totals':total,'model_test':summary['model_test'],'replay_wall_us':timing['wall_us'],
                      'cell_points':[(r['experiment'],r['suite'],r['payouts'],r['points']) for r in cells]},indent=2))


if __name__=='__main__':main()
