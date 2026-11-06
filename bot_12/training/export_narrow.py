"""Export additional range curves without changing the six-range API."""
import csv
import json
from pathlib import Path
import sys

def main():
    rows=list(csv.reader(open(sys.argv[1],newline='')))
    order=json.loads(Path('assets/preflop.json').read_text())['order']
    combos=lambda k:6 if k//13==k%13 else 4 if k//13>k%13 else 12
    thresholds=[14,34]
    data={'nominal_ranges':[.01,.025],'rank_thresholds':thresholds,
          'actual_combo_fractions':[sum(combos(k) for k in range(169) if order[k]<t)/1326 for t in thresholds],
          'range_definition':'same random-equity-ranked whole preflop classes as preflop.json',
          'samples_per_class_and_range':200000,'seed':1200501,
          'equities':[[float(x) for x in row[1:]] for row in sorted(rows,key=lambda row:int(row[0]))]}
    Path('assets/preflop_narrow.json').write_text(json.dumps(data,separators=(',',':')))
    print({k:v for k,v in data.items() if k!='equities'})
if __name__=='__main__':main()
