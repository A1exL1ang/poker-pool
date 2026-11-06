"""Export deterministic tree prefixes and assess speed/accuracy on validation.

No refitting or policy data is used. Prefix choice is model tuning, so the
validation data here must not be presented as an untouched final test.
"""
import argparse
import json
from pathlib import Path
import struct
import time
import zlib


def prefix(source, out, count):
    raw = zlib.decompress(Path(source).read_bytes())
    magic, trees, outputs = struct.unpack_from('<8sII', raw)
    if magic != b'BOT11EQ2' or not 0 < count <= trees:
        raise ValueError('requires compact model and valid tree budget')
    offset = 16
    chunks = [struct.pack('<8sII', magic, count, outputs)]
    for i in range(trees):
        nodes, leaves = struct.unpack_from('<II', raw, offset)
        end = offset + 8 + nodes*6 + leaves*outputs
        if i < count:
            chunks.append(raw[offset:end])
        offset = end
    Path(out).write_bytes(zlib.compress(b''.join(chunks),6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='assets/equity_64.bin.z')
    p.add_argument('--trees', type=int, nargs='+', default=[16,24,32,48,64])
    p.add_argument('--validation', default='benchmarks/equity_validation.npz')
    p.add_argument('--report', default='benchmarks/model_budgets.json')
    p.add_argument('--export-only', action='store_true')
    args = p.parse_args()
    for count in args.trees:
        prefix(args.source, f'assets/equity_{count}.bin.z', count)
    if args.export_only:
        return
    import numpy as np
    from pokerbot.equity import EquityModel, LABELS
    data = np.load(args.validation)
    xs = data['x'].tolist()
    report = {'validation_not_final_test': True, 'states':len(xs), 'budgets': {}}
    for count in args.trees:
        model = EquityModel(f'assets/equity_{count}.bin.z')
        start = time.monotonic()
        pred = np.asarray([model.predict(x) for x in xs])
        elapsed = time.monotonic()-start
        report['budgets'][str(count)] = {'bytes':Path(f'assets/equity_{count}.bin.z').stat().st_size,
            'mae':dict(zip(LABELS, abs(pred-data['y']).mean(axis=0).tolist())),
            'rmse':dict(zip(LABELS, np.sqrt(((pred-data['y'])**2).mean(axis=0)).tolist())),
            'inference_mean_us': elapsed/len(xs)*1e6}
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
