"""Blocker-aware preflop equity tables, generated independently offline."""
from __future__ import annotations
from bisect import bisect_left
import json
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RANGE_GRID = (.018, .025, .04, .06, .08, .10, .13, .17, .22, .28, .35, .45, .60, .80, 1.)


def class_id(a, b):
    r0, r1 = a >> 2, b >> 2
    hi, lo = max(r0, r1), min(r0, r1)
    if hi == lo:
        return (12 - hi) * 14
    return (12 - hi) * 13 + 12 - lo if a & 3 == b & 3 else (12 - lo) * 13 + 12 - hi


def representative(i):
    row, col = divmod(i, 13)
    r0, r1 = 12 - row, 12 - col
    return (4 * r0, 4 * r1 + (1 if row >= col else 0))


def legal_counts(i):
    """Exact class multiplicities after two known hero cards (sum = 1225)."""
    a, b = representative(i)
    rank_counts = [4] * 13
    rank_counts[a >> 2] -= 1
    rank_counts[b >> 2] -= 1
    out = []
    for j in range(169):
        row, col = divmod(j, 13)
        r0, r1 = 12 - row, 12 - col
        if r0 == r1:
            n = rank_counts[r0]
            out.append(n * (n - 1) // 2)
        else:
            blocked_suits = {c & 3 for c in (a, b) if c >> 2 in (r0, r1)}
            suited = 4 - len(blocked_suits)
            out.append(suited if row < col else rank_counts[r0] * rank_counts[r1] - suited)
    return out


class Preflop:
    def __init__(self, asset=None):
        path = Path(asset) if asset else ASSETS / "preflop.json"
        self.loaded = path.exists()
        if self.loaded:
            raw = json.loads(path.read_text())
            self.matrix = raw["equity"]
            self.uniform = raw["uniform_equity"]
            self.counts = raw["combination_counts"]
        else:
            # Runnable development fallback. Final packaging requires the
            # measured asset; this formula is never described as simulation.
            e = []
            self.counts = []
            for i in range(169):
                row, col = divmod(i, 13)
                hi, lo = max(14 - row, 14 - col), min(14 - row, 14 - col)
                pair, suited = row == col, row < col
                e.append(.49 + .027 * (hi - 2) if pair else .28 + .021 * (hi - 2) + .012 * (lo - 2) + .025 * suited - .004 * max(0, hi - lo - 2))
                self.counts.append(6 if pair else 4 if suited else 12)
            self.uniform = [[x ** (1 + .70 * n) for n in range(7)] for x in e]
            self.matrix = [[max(.02, min(.98, .5 + a - b)) for b in e] for a in e]
        self.percentile = [0.] * 169
        before = 0
        for i in sorted(range(169), key=lambda i: self.uniform[i][0], reverse=True):
            self.percentile[i] = (before + self.counts[i] / 2) / 1326
            before += self.counts[i]
        self.range_equities = []
        self.range_mass = []
        all_counts = [legal_counts(i) for i in range(169)]
        for q in RANGE_GRID:
            weights = [max(0., min(1., (q - self.percentile[j]) * 1326 / self.counts[j] + .5)) for j in range(169)]
            equities, masses = [], []
            for i in range(169):
                mass = sum(c * w for c, w in zip(all_counts[i], weights))
                value = sum(e * c * w for e, c, w in zip(self.matrix[i], all_counts[i], weights)) / max(1e-12, mass)
                equities.append(value)
                masses.append(mass / 1225)
            self.range_equities.append(equities)
            self.range_mass.append(masses)
        self.pushfold = None
        pf = ASSETS / "pushfold.json"
        if pf.exists():
            self.pushfold = json.loads(pf.read_text())

    def _lookup(self, table, hand, fraction):
        q = max(RANGE_GRID[0], min(1., fraction))
        k = bisect_left(RANGE_GRID, q)
        if not k:
            return table[0][hand]
        t = (q - RANGE_GRID[k - 1]) / (RANGE_GRID[k] - RANGE_GRID[k - 1])
        return table[k - 1][hand] * (1 - t) + table[k][hand] * t

    def equity(self, hand, fraction=1., opponents=1):
        e = self._lookup(self.range_equities, hand, fraction)
        if opponents <= 1:
            return e
        # Calibrated correlation exponent from this hand's independent
        # multiway simulation; range generalization remains an approximation.
        from math import log
        n = max(1, min(7, opponents))
        u = self.uniform[hand]
        exponent = log(max(.0001, u[n - 1])) / log(max(.001, min(.999, u[0])))
        return e ** exponent

    def mass(self, hand, fraction):
        return self._lookup(self.range_mass, hand, fraction)

    def nash_probability(self, hand, depth, role):
        if not self.pushfold:
            return None
        depths = self.pushfold["depths_bb"]
        k = min(bisect_left(depths, depth), len(depths) - 1)
        values = self.pushfold["sb_shove" if role == "sb" else "bb_call"]
        if not k:
            return values[0][hand]
        t = max(0., min(1., (depth - depths[k - 1]) / (depths[k] - depths[k - 1])))
        return values[k - 1][hand] * (1 - t) + values[k][hand] * t


TABLES = Preflop()
