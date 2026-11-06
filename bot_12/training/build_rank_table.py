"""Build original suit-free seven-card rank table for exact dry-river equity."""
from itertools import combinations_with_replacement
from pathlib import Path
import struct
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from equity import evaluate

def main():
    powers=[5**r for r in range(13)];records=[]
    for ranks in combinations_with_replacement(range(13),7):
        if any(ranks.count(r)>4 for r in set(ranks)):continue
        # Round-robin suits give distinct physical cards for each rank group
        # and at most two cards per suit, so no flush is possible.
        cards=[4*r+(i%4) for i,r in enumerate(ranks)]
        records.append((sum(powers[r] for r in ranks),evaluate(cards)))
    data=b''.join(struct.pack('<II',*r) for r in sorted(records))
    Path('assets/rank_scores.zlib').write_bytes(zlib.compress(data,9))
    print('records',len(records),'raw_bytes',len(data),'compressed_bytes',Path('assets/rank_scores.zlib').stat().st_size)

if __name__=='__main__':main()
