"""Bounded, dependency-free inference for independently generated equity assets."""
from array import array
import json
from pathlib import Path
import struct
import sys
import zlib

ASSETS = Path(__file__).resolve().parent.parent / 'assets'
LABELS = ('uniform1', 'uniform2', 'uniform3', 'pf40', 'pf15', 'pf05',
          'current35', 'current15', 'joint40_35', 'joint15_15')


def hand_key(hole):
    a, b = sorted(hole, reverse=True)
    ranks = '23456789TJQKA'
    h, l = ranks[a >> 2], ranks[b >> 2]
    return h + l + ('' if h == l else ('s' if a & 3 == b & 3 else 'o'))


class EquityModel:
    def __init__(self, path=None):
        self.trees = []
        path = Path(path) if path else ASSETS / 'equity.bin.z'
        raw = memoryview(zlib.decompress(path.read_bytes()))
        magic, count, outputs = struct.unpack_from('<8sII', raw)
        if magic not in (b'BOT11EQ1', b'BOT11EQ2') or outputs != 10 or not 1 <= count <= 256:
            raise ValueError('invalid equity model')
        self.compact = magic == b'BOT11EQ2'
        offset = 16
        def take(code, n):
            nonlocal offset
            a = array(code)
            size = a.itemsize * n
            a.frombytes(raw[offset:offset + size])
            if sys.byteorder != 'little' and a.itemsize > 1:
                a.byteswap()
            offset += size
            return a
        for _ in range(count):
            nodes, leaves = struct.unpack_from('<II', raw, offset)
            offset += 8
            fs = take('b', nodes)
            if magic == b'BOT11EQ2':
                ts = take('b', nodes)
                links = take('i', nodes)
                values = take('B', leaves * outputs)
                self.trees.append((fs, ts, links, values))
                continue
            ts = take('f', nodes)
            left = take('i', nodes)
            right = take('i', nodes)
            values = take('H', leaves * outputs)
            self.trees.append((fs, ts, left, right, values))
        if offset != len(raw):
            raise ValueError('unexpected trailing model data')

    def predict(self, x):
        if self.compact:
            # Fixed ten-output unrolling avoids hundreds of Python loop/index
            # operations. Integer accumulators preserve quantized values exactly.
            s0=s1=s2=s3=s4=s5=s6=s7=s8=s9=0
            for fs, ts, links, v in self.trees:
                n = 0
                f = fs[n]
                while f >= 0:
                    n = n+1 if x[f] <= ts[n] else links[n]
                    f = fs[n]
                b = (-links[n]-1)*10
                s0 += v[b]; s1 += v[b+1]; s2 += v[b+2]; s3 += v[b+3]; s4 += v[b+4]
                s5 += v[b+5]; s6 += v[b+6]; s7 += v[b+7]; s8 += v[b+8]; s9 += v[b+9]
            scale = 1.0 / (255 * len(self.trees))
            return [s0*scale,s1*scale,s2*scale,s3*scale,s4*scale,s5*scale,s6*scale,s7*scale,s8*scale,s9*scale]
        sums = [0] * 10
        for tree in self.trees:
            fs, ts, left, right, values = tree
            n = 0
            while fs[n] >= 0:
                n = left[n] if x[fs[n]] <= ts[n] else right[n]
            base = left[n] * 10
            for j in range(10):
                sums[j] += values[base + j]
        scale = 1.0 / (65535 * len(self.trees))
        return [s * scale for s in sums]


def load_preflop(path=None):
    return json.loads((Path(path) if path else ASSETS / 'preflop.json').read_text())

