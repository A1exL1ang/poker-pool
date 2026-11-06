"""Suit-invariant starting-hand ordering used to define synthetic opponent ranges.

This is a documented local range abstraction, not a solved equilibrium chart.
Percentiles use physical combination counts and include the whole boundary class.
"""
from bisect import bisect_left


def class_index(hole):
    a, b = hole[0] >> 2, hole[1] >> 2
    if a == b:
        return a * 13 + a
    hi, lo = max(a, b), min(a, b)
    return hi * 13 + lo if (hole[0] & 3) == (hole[1] & 3) else lo * 13 + hi


def class_hole(index):
    a, b = divmod(index, 13)
    if a == b:
        return (a * 4, a * 4 + 1)
    return (a * 4, b * 4) if a > b else (a * 4, b * 4 + 1)


def hand_score(hole):
    a, b = hole[0] // 4 + 2, hole[1] // 4 + 2
    hi, lo = max(a, b), min(a, b)
    if hi == lo:
        return 0.49 + 0.028 * (hi - 2)
    return (0.25 + 0.0205 * hi + 0.009 * lo
            + 0.025 * ((hole[0] & 3) == (hole[1] & 3))
            - 0.0035 * max(hi - lo - 1, 0) + 0.012 * (hi == 14))


CLASS_ORDER = sorted(range(169), key=lambda i: (-hand_score(class_hole(i)), i))
CLASS_PERCENTILES = [0.0] * 169
CLASS_UPPER = [0.0] * 169
_count = 0
for _index in CLASS_ORDER:
    _a, _b = divmod(_index, 13)
    _combos = 6 if _a == _b else 4 if _a > _b else 12
    CLASS_PERCENTILES[_index] = (_count + 0.5 * _combos) / 1326
    _count += _combos
    CLASS_UPPER[_index] = _count / 1326


def preflop_percentile(hole):
    """Combination-weighted midrank; lower means stronger."""
    return CLASS_PERCENTILES[class_index(hole)]


RANGE_FRACTIONS = (0.06, 0.12, 0.25, 0.45, 1.0)


def range_weights(fraction):
    fraction = max(RANGE_FRACTIONS[0], min(1.0, fraction))
    i = bisect_left(RANGE_FRACTIONS, fraction)
    if i == 0:
        return 0, 0, 0.0
    if i == len(RANGE_FRACTIONS):
        return i - 1, i - 1, 0.0
    lo = i - 1
    weight = (fraction - RANGE_FRACTIONS[lo]) / (RANGE_FRACTIONS[i] - RANGE_FRACTIONS[lo])
    return lo, i, weight
