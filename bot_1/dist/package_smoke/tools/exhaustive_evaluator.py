"""Exhaust all 2,598,960 five-card hands and check category frequencies."""
import itertools
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pokerbot.evaluator import evaluate_bits,BITS

started=time.perf_counter()
counts=[0]*9
for cards in itertools.combinations(range(52),5):
    bits=BITS[cards[0]]|BITS[cards[1]]|BITS[cards[2]]|BITS[cards[3]]|BITS[cards[4]]
    counts[evaluate_bits(bits)>>24]+=1
expected=[1302540,1098240,123552,54912,10200,5108,3744,624,40]
result={"hands":sum(counts),"category_counts_low_to_high":counts,"expected":expected,"passed":counts==expected,"elapsed_seconds":time.perf_counter()-started}
Path("benchmarks/exhaustive_evaluator.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
assert counts==expected
