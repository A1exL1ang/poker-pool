"""Lossless integer split packing + uint8 leaf quantization of original model.

Tree splits are exact because all input features are integers. Leaf values
have <=1/510 added absolute error, before ensemble averaging. Training only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import zlib
import numpy as np
from pokerbot.equity import EquityModel, LABELS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='assets/equity.bin.z')
    p.add_argument('--out', default='assets/equity_compact.bin.z')
    p.add_argument('--validation', default='benchmarks/equity_validation.npz')
    args = p.parse_args()
    model = EquityModel(args.source)
    chunks = [struct.pack('<8sII', b'BOT11EQ2', len(model.trees), 10)]
    for fs, ts, left, right, vals in model.trees:
        fs = np.asarray(fs, dtype='i1')
        ts = np.asarray(ts)
        left = np.asarray(left)
        right = np.asarray(right)
        leaf = fs < 0
        if not np.all(left[~leaf] == np.arange(len(left))[~leaf] + 1):
            raise ValueError('expected depth-first preordered trees')
        thresholds = np.where(leaf, 0, np.floor(ts))
        if thresholds.min() < -128 or thresholds.max() > 127:
            raise ValueError('threshold exceeds integer encoding')
        thresholds = thresholds.astype('i1')
        links = np.where(leaf, -left-1, right).astype('<i4')
        quant = np.rint(np.asarray(vals, dtype='f8')/257).astype('u1')
        chunks.extend([struct.pack('<II', len(left), len(vals)//10), fs.tobytes(), thresholds.tobytes(), links.tobytes(), quant.tobytes()])
    raw = b''.join(chunks)
    Path(args.out).write_bytes(zlib.compress(raw, 6))
    del model, chunks, raw
    packed = EquityModel(args.out)
    data = np.load(args.validation)
    started = time.monotonic()
    pred = np.asarray([packed.predict(x) for x in data['x'].tolist()])
    elapsed = time.monotonic() - started
    original = data['predicted']
    report = {'format':'BOT11EQ2','trees':len(packed.trees),'bytes':Path(args.out).stat().st_size,
              'sha256':hashlib.sha256(Path(args.out).read_bytes()).hexdigest(),
              'validation_states':len(pred),'max_quantization_delta':float(abs(pred-original).max()),
              'mean_quantization_delta':float(abs(pred-original).mean()),
              'mae':dict(zip(LABELS, abs(pred-data['y']).mean(axis=0).tolist())),
              'inference_mean_us':elapsed/len(pred)*1e6}
    Path('benchmarks/equity_compaction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
