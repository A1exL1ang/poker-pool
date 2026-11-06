"""Export a compact, versioned binary forest; no pickle/executable payloads."""
import argparse
from array import array
import gzip
import json
import math
from pathlib import Path
import struct
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import equity_features


def export(source, out, count):
    with gzip.open(source, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    trees = data['trees'][:count]
    meta = dict(data['metadata'])
    meta.update(format='bot9_forest_v1_le', trees=len(trees), source_trees=data['metadata']['trees'])
    outputs = len(meta['targets'])
    blob = json.dumps(meta, separators=(',', ':')).encode()
    with Path(out).open('wb') as f:
        f.write(b'B9EQv1\0\0')
        f.write(struct.pack('<II', len(blob), len(trees)))
        f.write(blob)
        for tree in trees:
            n = len(tree)
            features = bytearray(n)
            threshold = array('f', [0.]) * n
            right = array('I', [0]) * n
            values = array('f', [0.]) * (n*outputs)
            for i, node in enumerate(tree):
                features[i] = node[0] if node[0] >= 0 else 255
                if node[0] >= 0:
                    assert node[2] == i+1
                    # Features are integer-valued. Flooring preserves every
                    # branch while float32 rounding near integers may not.
                    threshold[i] = math.floor(node[1])
                    right[i] = node[3]
                else:
                    values[i*outputs:i*outputs+outputs] = array('f', node[1])
            f.write(struct.pack('<I', n))
            f.write(features)
            for a in (threshold, right, values):
                if sys.byteorder != 'little':
                    a.byteswap()
                a.tofile(f)
    return meta


def evaluate(path, holdout, max_rows=6000):
    from model import EquityModel
    import csv
    model = EquityModel(path)
    with open(holdout, newline='') as f:
        keys = next(csv.reader(f))[7:]
    outputs = len(keys)
    err = [0.]*outputs
    squared = [0.]*outputs
    by_street = {n: [0.]*(outputs+1) for n in (3,4,5)}
    ns = []
    n = 0
    with open(holdout, newline='') as f:
        for row in csv.DictReader(f):
            h = tuple(int(row[k]) for k in ('h0','h1'))
            b = tuple(int(row[k]) for k in ('b0','b1','b2','b3','b4') if int(row[k]) >=0)
            t = time.perf_counter_ns()
            p = model.predict(h,b)
            ns.append(time.perf_counter_ns()-t)
            target = [float(row[k]) for k in keys]
            for i in range(outputs):
                d = p[i]-target[i]
                err[i]+=abs(d)
                squared[i]+=d*d
                by_street[len(b)][i]+=abs(d)
            by_street[len(b)][outputs]+=1
            n+=1
            if n >= max_rows:
                break
    ns.sort()
    return {'rows':n,'mae':[e/n for e in err],'rmse':[(e/n)**.5 for e in squared],
        'by_street': {str(k): [v/count for v in sums[:outputs]] for k,sums in by_street.items() if (count:=sums[outputs])},
        'predict_timing_ns': {'p50':ns[n//2],'p99':ns[int(n*.99)],'max':ns[-1], 'over1ms':sum(t>1000000 for t in ns)}}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='assets/equity_model.json.gz')
    p.add_argument('--out',default='assets/equity_model.bin')
    p.add_argument('--trees',type=int,default=20)
    p.add_argument('--holdout',default='measurements/equity_holdout.csv')
    p.add_argument('--rows',type=int,default=6000)
    p.add_argument('--report',default='measurements/compact_model.json')
    a=p.parse_args()
    meta=export(a.source,a.out,a.trees)
    report=evaluate(a.out,a.holdout,a.rows)
    report.update(trees=meta['trees'],bytes=Path(a.out).stat().st_size)
    Path(a.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
