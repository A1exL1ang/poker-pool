"""Standard-library export of the six original preflop range curves."""
import csv
import json
from pathlib import Path
import sys

def main():
    rows=list(csv.reader(open(sys.argv[1],newline='')))
    order=[None]*169
    for line in Path(sys.argv[2]).read_text().splitlines():
        k,e,r=line.split();order[int(k)]=int(r)
    data={'ranges':[.05,.10,.20,.35,.50,1.],
          'range_definition':'top hand classes ranked by independently estimated HU equity versus uniform cards, weighted by combo count',
          'equities':[[float(x) for x in row[1:]] for row in sorted(rows,key=lambda row:int(row[0]))],
          'order':order}
    Path('assets').mkdir(exist_ok=True)
    Path('assets/preflop.json').write_text(json.dumps(data,separators=(',',':')))
if __name__=='__main__':main()
