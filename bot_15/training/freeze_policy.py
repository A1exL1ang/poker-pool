"""Freeze the highest predeclared tuning objective; never reads held-out data."""
import hashlib
import json
from pathlib import Path

from benchmarks.tournament import summary


def main():
    source = Path('artifacts/policy_refinement.json')
    result = json.loads(source.read_text())
    winner = result['ranking'][0]
    reference = 'open1.0_pressure1.0'
    selected_rows = {(r['population'], r['payout']): r['rows'] for r in result['results'] if r['candidate'] == winner['candidate']}
    reference_rows = {(r['population'], r['payout']): r['rows'] for r in result['results'] if r['candidate'] == reference}
    differences = []
    for key, rows in selected_rows.items():
        other = reference_rows[key]
        assert [r['seed'] for r in rows] == [r['seed'] for r in other]
        differences.extend((a['points'] - b['points']) * 8 for a, b in zip(rows, other))
    policy = dict(status='frozen before final held-out evaluation',
                  candidate=winner['candidate'], config=winner['config'],
                  selection_source=str(source), selection_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  objective=result['objective'], tuning_objective=winner['objective'],
                  paired_tuning_difference_vs_untuned=summary(differences),
                  limitation='Selected among ten refinement configurations on synthetic opponents/payouts; tuning estimate is selection-biased and not a contest prediction.')
    Path('artifacts/policy.json').write_text(json.dumps(policy, indent=2))
    print(json.dumps(policy, indent=2))


if __name__ == '__main__':
    main()
