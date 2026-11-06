"""Portable multi-output regression forest inference, stdlib only at gameplay."""
import gzip
import json
import struct
import sys
from array import array
from pathlib import Path
from features import equity_features

_CACHE = {}


class EquityModel:
    def __init__(self, path=None):
        assets = Path(__file__).with_name('assets')
        path = Path(path) if path else assets / ('equity_model.bin' if (assets/'equity_model.bin').exists() else 'equity_model.json.gz')
        self.trees = []
        self.metadata = {}
        if path.exists():
            stat = path.stat()
            key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
            data = _CACHE.get(key)
            if data is None:
                if path.suffix == '.bin':
                    data = self._load_binary(path)
                else:
                    with gzip.open(path, 'rt', encoding='utf-8') as f:
                        data = json.load(f)
                _CACHE[key] = data
            self.trees = data['trees']
            self.metadata = data['metadata']
        self.binary = bool(self.metadata.get('format') == 'bot9_forest_v1_le')
        self.outputs = len(self.metadata.get('targets', [0, 1, 2, 3]))

    @staticmethod
    def _load_binary(path):
        with open(path, 'rb') as f:
            if f.read(8) != b'B9EQv1\0\0':
                raise ValueError('Invalid bot_9 forest header')
            meta_len, count = struct.unpack('<II', f.read(8))
            if meta_len > 1000000 or count > 256:
                raise ValueError('Invalid forest dimensions')
            metadata = json.loads(f.read(meta_len))
            outputs = len(metadata['targets'])
            if outputs not in (4, 6):
                raise ValueError('Unsupported forest outputs')
            trees = []
            for _ in range(count):
                n, = struct.unpack('<I', f.read(4))
                if n > 1000000:
                    raise ValueError('Invalid tree dimensions')
                features = f.read(n)
                arrays = []
                for code, width in (('f',n), ('I',n), ('f',n*outputs)):
                    a = array(code)
                    a.frombytes(f.read(width*4))
                    if sys.byteorder != 'little':
                        a.byteswap()
                    arrays.append(a)
                trees.append((features, *arrays))
            return {'metadata':metadata, 'trees':trees}

    def predict(self, hole, board):
        x = equity_features(hole, board)
        if not self.trees:
            return self._fallback(x)
        if self.binary:
            return self._predict_binary(x)
        a = b = c = d = e = f = 0.0
        for tree in self.trees:
            i = 0
            while tree[i][0] >= 0:
                node = tree[i]
                i = node[2] if x[node[0]] <= node[1] else node[3]
            values = tree[i][1]
            a += values[0]
            b += values[1]
            c += values[2]
            d += values[3]
            if self.outputs == 6:
                e += values[4]
                f += values[5]
        n = len(self.trees)
        if self.outputs == 6:
            return a/n, b/n, c/n, d/n, e/n, f/n
        return a / n, b / n, c / n, d / n

    def _predict_binary(self, x):
        a = b = c = d = e = f = 0.
        outputs = self.outputs
        for features, thresholds, right, values in self.trees:
            i = 0
            feature = features[i]
            while feature != 255:
                i = i+1 if x[feature] <= thresholds[i] else right[i]
                feature = features[i]
            j = i*outputs
            a += values[j]
            b += values[j+1]
            c += values[j+2]
            d += values[j+3]
            if outputs == 6:
                e += values[j+4]
                f += values[j+5]
        n = len(self.trees)
        if outputs == 6:
            return a/n, b/n, c/n, d/n, e/n, f/n
        return a/n, b/n, c/n, d/n

    @staticmethod
    def _fallback(x):
        category = x[6]
        eq = (0.22, 0.56, 0.78, 0.87, 0.91, 0.95, 0.98, 0.995, 0.999)[category]
        if category <= 1:
            eq += (x[7] - 9) * 0.025
        return tuple(max(0.02, min(0.999, eq - p)) for p in (0, .06, .11, .17))
