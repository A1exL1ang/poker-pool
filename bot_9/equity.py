"""Offline preflop equity lookup, independent Monte Carlo, seed 901.

assets/preflop.csv has 50,000 rollouts per cell (random and 3 range models).
Ranges comprise complete hand types ranked by random heads-up equity; actual
combination fractions are 536/1326, 208/1326, 68/1326. This is a range model,
not a solved strategy. No gameplay simulation or external services are used.
"""
import csv
import math
from pathlib import Path


def hand_type(hole):
    """169 grid: upper triangle suited, lower triangle offsuit."""
    a, b = hole
    r, s = a >> 2, b >> 2
    if r < s:
        r, s = s, r
    return r * 13 + s if r == s or (a & 3) == (b & 3) else s * 13 + r


_table = [None] * 169
with (Path(__file__).resolve().parent / 'assets' / 'preflop.csv').open(newline='') as _file:
    for _row in csv.DictReader(_file):
        _table[hand_type((int(_row['h0']), int(_row['h1'])))] = tuple(
            float(_row[key]) for key in ('eq_random', 'eq_top40', 'eq_top15', 'eq_top5'))
if any(value is None for value in _table):
    raise ValueError('Incomplete offline preflop equity asset')
PREFLOP = tuple(_table)
_percentiles = [0.0] * 169
_mass = 0
for _type in sorted(range(169), key=lambda t: PREFLOP[t][0], reverse=True):
    _r, _s = divmod(_type, 13)
    _weight = 6 if _r == _s else 4 if _r > _s else 12
    _percentiles[_type] = (_mass + _weight / 2) / 1326
    _mass += _weight
PERCENTILE = tuple(_percentiles)
RANGE_FRACTIONS = (1.0, 536 / 1326, 208 / 1326, 68 / 1326)
_LOG_FRACTIONS = tuple(math.log(p) for p in RANGE_FRACTIONS)


def preflop_percentile(hole):
    """Combination-weighted strength percentile; 0 strongest, 1 weakest."""
    return PERCENTILE[hand_type(hole)]


def equity_vs_range(hole, top_fraction=1.0):
    """Heads-up equity, interpolated in log range size; clamps beyond table.

    Fractions below ~5.1% use the tightest modeled range. Fractions above 1
    use uniformly random holdings. Interpolated values are approximations.
    """
    values = PREFLOP[hand_type(hole)]
    if top_fraction >= 1:
        return values[0]
    if top_fraction <= RANGE_FRACTIONS[-1]:
        return values[-1]
    fraction_log = math.log(top_fraction)
    for i in range(3):
        if top_fraction >= RANGE_FRACTIONS[i + 1]:
            t = ((_LOG_FRACTIONS[i] - fraction_log) /
                 (_LOG_FRACTIONS[i] - _LOG_FRACTIONS[i + 1]))
            return values[i] + t * (values[i + 1] - values[i])
    return values[-1]


def preflop_equity(hole, opponents=1):
    """Random heads-up equity; power approximation if opponents >1.

    Multiway outcomes are correlated: this is not exact multiway equity.
    """
    equity = PREFLOP[hand_type(hole)][0]
    return equity ** opponents if opponents != 1 else equity
