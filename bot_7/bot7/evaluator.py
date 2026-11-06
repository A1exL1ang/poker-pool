"""Exact 5-7 card evaluation with a generated rank-multiset lookup.

Larger scores win. Bits 20+ encode category (0 high card through 8 straight
flush); five rank nibbles break ties. No third-party runtime dependencies.
"""

import gzip
import struct
from pathlib import Path

RANK_MASK = (1 << 39) - 1
BIAS = sum(3 << (39 + 4 * s) for s in range(4))
FLUSH_FLAGS = sum(8 << (39 + 4 * s) for s in range(4))
ENC = tuple((1 << (3 * (c // 4))) + (1 << (39 + 4 * (c % 4)))
            + (1 << (55 + 13 * (c % 4) + c // 4)) for c in range(52))

with gzip.open(Path(__file__).resolve().parent.parent / "assets" / "eval.bin.gz", "rb") as f:
    _data = f.read()
_n = struct.unpack_from("<I", _data)[0]
RANKS = dict(struct.iter_unpack("<QI", _data[4:4 + 12 * _n]))
FLUSH = struct.unpack_from("<8192I", _data, 4 + 12 * _n)
del _data, _n


def evaluate_sum(total):
    flags = (total + BIAS) & FLUSH_FLAGS
    if flags:
        suit = (flags.bit_length() - 43) // 4
        return FLUSH[(total >> (55 + 13 * suit)) & 8191]
    return RANKS[total & RANK_MASK]


def evaluate(cards):
    return evaluate_sum(sum(ENC[c] for c in cards))

