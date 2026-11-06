"""Evaluate a frozen runtime model on independent binary MC labels (stdlib)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time
from pokerbot.equity import EquityModel, LABELS
from pokerbot.features import features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='benchmarks/equity_final_test.json')
    args = p.parse_args()
    model = EquityModel()
    raw = Path(args.data).read_bytes()
    if len(raw)%48:
        raise ValueError('incomplete fixed-length record')
    totals = {n:{'count':0,'abs':[0.]*10,'sq':[0.]*10} for n in ('all',3,4,5)}
    start = time.monotonic()
    for record in struct.iter_unpack('<8B10f',raw):
        count = record[0]
        pred = model.predict(features(record[1:3], record[3:3+count]))
        truth = record[8:]
        for group in ('all',count):
            sums = totals[group]
            sums['count'] += 1
            for j in range(10):
                err = pred[j]-truth[j]
                sums['abs'][j] += abs(err)
                sums['sq'][j] += err*err
    report = {'independent_final_test':True,'states':len(raw)//48,'trees':len(model.trees),
              'data_sha256':hashlib.sha256(raw).hexdigest(),
              'model_sha256':hashlib.sha256(Path('assets/equity.bin.z').read_bytes()).hexdigest(),
              'seconds':time.monotonic()-start,'metrics':{}}
    for group,sums in totals.items():
        report['metrics'][str(group)]={'count':sums['count'],
            'mae':dict(zip(LABELS,(s/sums['count'] for s in sums['abs']))),
            'rmse':dict(zip(LABELS,(math.sqrt(s/sums['count']) for s in sums['sq'])))}
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
