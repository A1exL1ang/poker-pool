"""Original deep-stack opening order emphasizing domination and playability.

These are heuristic ordering scores, not solver frequencies. Short-stack
shoves continue to use all-in equity ordering, where realization differs.
"""
from .equity import DATA, PERCENTILE
from .cards import RANKS


def opening_score(name):
    high, low = RANKS.index(name[0]) + 2, RANKS.index(name[1]) + 2
    if high == low:
        return {2: .22, 3: .19, 4: .16, 5: .13, 6: .10, 7: .08, 8: .065,
                9: .045, 10: .032, 11: .022, 12: .012, 13: .006, 14: 0.}[high]
    suited = name[-1] == "s"
    if high == 14:
        return ({13: .025, 12: .045, 11: .080, 10: .10, 9: .12, 8: .14,
                 7: .16, 6: .18, 5: .12, 4: .17, 3: .20, 2: .22} if suited else
                {13: .035, 12: .065, 11: .13, 10: .20, 9: .29, 8: .34,
                 7: .40, 6: .45, 5: .43, 4: .47, 3: .50, 2: .53})[low]
    if high == 13:
        return ({12: .055, 11: .10, 10: .14, 9: .22, 8: .28, 7: .32,
                 6: .36, 5: .38, 4: .40, 3: .44, 2: .47} if suited else
                {12: .11, 11: .19, 10: .27, 9: .40, 8: .49, 7: .56,
                 6: .62, 5: .64, 4: .66, 3: .68, 2: .70})[low]
    if high == 12:
        return ({11: .11, 10: .17, 9: .25, 8: .34, 7: .42, 6: .48,
                 5: .52, 4: .55, 3: .57, 2: .60} if suited else
                {11: .22, 10: .33, 9: .50, 8: .61, 7: .68, 6: .74,
                 5: .77, 4: .80, 3: .83, 2: .86})[low]
    if high == 11:
        return ({10: .14, 9: .24, 8: .34, 7: .48, 6: .57, 5: .63,
                 4: .67, 3: .70, 2: .73} if suited else
                {10: .33, 9: .48, 8: .62, 7: .73, 6: .80, 5: .84,
                 4: .88, 3: .91, 2: .94})[low]
    gap = high - low
    if suited and gap == 1:
        return {10: .20, 9: .25, 8: .31, 7: .33, 6: .36, 5: .39, 4: .53, 3: .65}[high]
    if suited and gap == 2:
        return {10: .30, 9: .38, 8: .42, 7: .45, 6: .48, 5: .52, 4: .65}[high]
    if suited and gap == 3:
        return .45 + (10 - high) * .045
    return .50 + PERCENTILE[name] * .50 + (.10 if not suited else 0)


_order = sorted(DATA["order"], key=opening_score)
OPEN_PERCENTILE = {}
_n = 0
for _name in _order:
    _count = 6 if len(_name) == 2 else 4 if _name[-1] == "s" else 12
    OPEN_PERCENTILE[_name] = (_n + _count / 2) / 1326
    _n += _count
