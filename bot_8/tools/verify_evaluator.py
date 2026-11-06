"""Exhaustive five-card category counts, independently known combinatorial totals."""
import itertools
import json
import time
from kestrel.cards import evaluate

def main():
    expected = [1302540,1098240,123552,54912,10200,5108,3744,624,40]
    counts = [0]*9
    start=time.perf_counter()
    for cards in itertools.combinations(range(52),5):
        counts[evaluate(cards)>>20]+=1
    result={"hands":sum(counts),"counts":counts,"expected":expected,"passed":counts==expected,"seconds":time.perf_counter()-start}
    print(json.dumps(result,indent=2))
    if counts != expected:
        raise SystemExit(1)

if __name__=="__main__":
    main()
