"""Convert narrow preflop Monte Carlo CSV into a runtime asset."""
import argparse
import csv
import json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--input', default='training/preflop_narrow.csv')
p.add_argument('--output', default='assets/preflop_narrow.json')
p.add_argument('--samples', type=int, default=25000)
p.add_argument('--seed', type=int, default=161005)
a = p.parse_args()
rows = list(csv.reader(open(a.input)))
assert len(rows) == 169 and all(int(r[0]) == i and len(r) == 7 for i, r in enumerate(rows))
data = {'description': 'Original Monte Carlo equity vs explicit premium reraising ranges. n>3 extrapolated at runtime.',
        'range_fractions': [.025, .05], 'ranges': ['QQ+,AK (34 combos)', 'TT+,AQ+,AJs,KQs (70 combos)'],
        'samples_per_entry': a.samples, 'seed': a.seed, 'max_opponents': 3,
        'equity': [[float(x) for x in r[1:]] for r in rows]}
Path(a.output).write_text(json.dumps(data, separators=(',', ':')))
