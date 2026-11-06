"""Offline tables plus a bounded, range-weighted turn/river estimator."""
from array import array
from bisect import bisect_left
from itertools import permutations
import json
from pathlib import Path
import random
import struct
import sys

from .cards import evaluate, hand_index, BoardEvaluator

ASSETS = Path(__file__).resolve().parent.parent / "assets"
with (ASSETS / "preflop.json").open() as _f:
    _PRE = json.load(_f)
PERCENTILE = tuple(_PRE["percentile"])
PREFLOP = tuple(tuple(row) for row in _PRE["equity"])
_TIGHT = None
if (ASSETS / "tight_preflop.json").exists():
    with (ASSETS / "tight_preflop.json").open() as _f:
        _TIGHT = tuple(tuple(row) for row in json.load(_f)["equity"])
WIDTHS = (0.15, 0.35, 0.65, 1.0)
RANGE_WIDTHS = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.0)
COMBOS = tuple((a, b) for a in range(52) for b in range(a + 1, 52))
RANGES = tuple(tuple((a, b) for a, b in COMBOS if PERCENTILE[hand_index(a, b)] <= width) for width in RANGE_WIDTHS)
_ALL_PERMS = tuple(permutations(range(4)))
_SUITED_PERMS = tuple(p for p in _ALL_PERMS if p[0] == 0)
_OFF_PERMS = tuple(p for p in _ALL_PERMS if p[0] == 0 and p[1] == 1)
_PAIR_PERMS = tuple(p for p in _ALL_PERMS if set(p[:2]) == {0, 1})


def interpolate(values, width):
    if width <= WIDTHS[0]:
        return values[0]
    for i in range(1, 4):
        if width <= WIDTHS[i]:
            f = (width - WIDTHS[i - 1]) / (WIDTHS[i] - WIDTHS[i - 1])
            return values[i - 1] * (1 - f) + values[i] * f
    return values[-1]


def preflop_equity(hole, width):
    h = hand_index(*hole)
    if _TIGHT is not None and width < 0.15:
        v0, v1 = _TIGHT[h]
        if width <= 0.03:
            return v0
        if width < 0.07:
            f = (width - 0.03) / 0.04
            return v0 * (1 - f) + v1 * f
        f = (width - 0.07) / 0.08
        return v1 * (1 - f) + PREFLOP[h][0] * f
    eq = interpolate(PREFLOP[h], width)
    if width < 0.15:
        # Until a tighter-range table is supplied, use a transparent conservative
        # correction. AA remains the preflop nuts; small pairs lose less equity
        # from range narrowing than dominated broadway hands.
        hi, lo = max(c >> 2 for c in hole), min(c >> 2 for c in hole)
        factor = min(1.0, (0.15 - width) / 0.12)
        penalty = 0.0 if hi == lo == 12 else 0.08 if hi == lo else 0.13
        eq -= factor * penalty
    return max(0.02, min(0.98, eq))


def flop_key(hole, board):
    a, b = sorted(hole, reverse=True)
    mapping = [-1] * 4
    mapping[a & 3] = 0
    if b & 3 != a & 3:
        mapping[b & 3] = 1
    nxt = max(mapping) + 1
    for i in range(4):
        if mapping[i] < 0:
            mapping[i] = nxt
            nxt += 1
    cards = tuple((c & ~3) | mapping[c & 3] for c in board)
    ps = _PAIR_PERMS if a >> 2 == b >> 2 else _SUITED_PERMS if a & 3 == b & 3 else _OFF_PERMS
    best = 1 << 18
    for p in ps:
        x, y, z = sorted((c & ~3) | p[c & 3] for c in cards)
        k = (x << 12) | (y << 6) | z
        if k < best:
            best = k
    return (hand_index(a, b) << 18) | best


class Equity:
    """Load once, outside decision timing. Assets are optional, never fetched."""
    def __init__(self, path=None, turn_samples=40, river_samples=80):
        self.keys = array("I")
        self.values = b""
        self.turn_samples = turn_samples
        self.river_samples = river_samples
        self.cache = {}
        self.asset_error = None
        path = Path(path) if path is not None else ASSETS / "flop.bin"
        if path.exists():
            try:
                with path.open("rb") as f:
                    header = f.read(16)
                    magic, count, self.trials = struct.unpack("<8sII", header)
                    if magic != b"B5FLOP01" or path.stat().st_size != 16 + count * 8:
                        raise ValueError("incomplete or invalid flop table")
                    self.keys.frombytes(f.read(count * 4))
                    if sys.byteorder != "little":
                        self.keys.byteswap()
                    self.values = f.read(count * 4)
            except (OSError, ValueError, struct.error) as exc:
                self.keys = array("I")
                self.asset_error = str(exc)

    def reset(self):
        self.cache.clear()

    def estimate(self, hole, board, width=1.0, pressure=0):
        if not board:
            return preflop_equity(hole, width)
        bucket = min(range(len(RANGE_WIDTHS)), key=lambda i: abs(RANGE_WIDTHS[i] - width))
        cache_key = (tuple(hole), tuple(board), bucket, pressure)
        if cache_key in self.cache:
            return self.cache[cache_key]
        if len(board) == 3 and self.keys:
            key = flop_key(hole, board)
            index = bisect_left(self.keys, key)
            if index < len(self.keys) and self.keys[index] == key:
                offset = index * 4
                values = tuple(v / 255 for v in self.values[offset:offset + 4])
                eq = interpolate(values, width)
                # Tables are conditional on preflop range, not future betting.
                # Betting updates are deliberately conservative and explicit.
                if pressure:
                    category = evaluate(hole + board) >> 20
                    penalty = (0.045, 0.085, 0.14)[min(pressure, 3) - 1]
                    if category >= 3:
                        penalty *= 0.25
                    eq = max(0.02, eq - penalty * (1.15 - eq))
                self.cache[cache_key] = eq
                return eq
        eq = self._sample(hole, board, bucket, pressure)
        if len(self.cache) > 32:
            self.cache.clear()
        self.cache[cache_key] = eq
        return eq

    def _sample(self, hole, board, bucket, pressure):
        known = set(hole + board)
        deck = tuple(c for c in range(52) if c not in known)
        combos = RANGES[bucket]
        # Common random numbers keep repeated states stable and comparisons
        # between action sizes from being dominated by Monte Carlo noise.
        seed = 0xB0505
        for c in sorted(hole) + board:
            seed = ((seed * 53) ^ (c + 1)) & 0xFFFFFFFF
        rng = random.Random(seed)
        choose = rng.randrange
        n = self.river_samples if len(board) == 5 else self.turn_samples
        evaluator = BoardEvaluator(board)
        hero_now = evaluator.pair(*hole)
        river = len(board) == 5
        board_high = max(c >> 2 for c in board)
        board_ranks = {c >> 2 for c in board}
        board_counts = [mask.bit_count() for mask in evaluator.suits]
        futures = {}
        wins = total = 0.0
        made_board = evaluate(board) if river else 0
        for _ in range(n):
            # Bounded rejection, including pathological blocker-heavy ranges.
            a = b = -1
            for _attempt in range(8):
                a, b = combos[choose(len(combos))]
                if a not in known and b not in known:
                    break
            else:
                a = deck[choose(len(deck))]
                b = deck[(deck.index(a) + 1 + choose(len(deck) - 1)) % len(deck)]
            weight = 1.0
            if pressure:
                vnow = evaluator.pair(a, b)
                cat = vnow >> 20
                ra, rb = a >> 2, b >> 2
                own_pair = ra == rb
                strong_pair = ra == board_high or rb == board_high or own_pair and ra >= board_high
                if cat >= 2 and (not river or vnow > made_board):
                    weight = 1.0
                elif strong_pair:
                    weight = 0.82
                elif cat == 1 and (not river or vnow > made_board):
                    weight = 0.38
                else:
                    weight = 0.18
                if not river and weight < 0.7:
                    sa, sb = a & 3, b & 3
                    if board_counts[sa] + 1 + (sa == sb) >= 4 or board_counts[sb] + 1 >= 4:
                        weight = max(weight, 0.55)
                if pressure >= 2:
                    weight *= weight
                if pressure >= 3:
                    weight *= 0.6 if weight < 0.5 else 1.0
            if river:
                hv = hero_now
                vv = vnow if pressure else evaluator.pair(a, b)
            else:
                runout = []
                while len(runout) < 5 - len(board):
                    c = deck[choose(len(deck))]
                    if c != a and c != b and c not in runout:
                        runout.append(c)
                runout_key = tuple(runout)
                future = futures.get(runout_key)
                if future is None:
                    future = evaluator
                    for c in runout:
                        future = future.add(c)
                    futures[runout_key] = future
                hv = future.pair(*hole)
                vv = future.pair(a, b)
            wins += weight * (1.0 if hv > vv else 0.5 if hv == vv else 0.0)
            total += weight
        return wins / total if total else 0.5
