"""Offline heads-up equity estimates. Runtime uses only the Python standard library.

Ranges mean the strongest fraction of starting combinations ranked by simulated
heads-up equity, not equilibrium ranges or ranges conditioned on postflop bets.
"""
import bisect
import gc
import gzip
import json
import math
from pathlib import Path

from .evaluator import evaluate

_ASSETS = Path(__file__).resolve().parent.parent / 'artifacts'
_PF = None
_TREES = None
_VALIDATED = False
TREE_LIMIT = 8
_FRACS = (.07, .2, .5, 1.0)


def load():
    global _PF, _TREES
    if _PF is None and (_ASSETS / 'equity_preflop.json').exists():
        _PF = json.loads((_ASSETS / 'equity_preflop.json').read_text())
    if _TREES is None and (_ASSETS / 'equity_model.json.gz').exists():
        with gzip.open(_ASSETS / 'equity_model.json.gz', 'rt') as stream:
            _TREES = json.load(stream)['trees']


def require_assets():
    """Validate deployment assets once, outside the timed decision method."""
    global _VALIDATED, _TREES
    if _VALIDATED:
        return
    load()
    if _PF is None or _TREES is None:
        raise RuntimeError('Equity assets missing: package artifacts/equity_preflop.json and artifacts/equity_model.json.gz')
    if len(_PF.get('percentiles', ())) != 169 or len(_PF.get('equities', ())) != 169 or not _TREES:
        raise RuntimeError('Equity assets have an invalid shape')
    if _PF.get('fractions') != [.03, .07, .12, .2, .35, .5, .75, 1]:
        raise RuntimeError('Preflop range fractions are invalid')
    if any(not 0 <= p <= 1 for p in _PF['percentiles']) or any(len(row) != 8 or any(not 0 <= q <= 1 for q in row) for row in _PF['equities']):
        raise RuntimeError('Preflop probabilities are invalid')
    for tree in _TREES:
        if len(tree) != 5 or not tree[0] or not (len(tree[0]) == len(tree[1]) == len(tree[2]) == len(tree[3])):
            raise RuntimeError('Equity tree arrays have an invalid shape')
        fs, ts, ls, rs, vs = tree
        n = len(fs)
        if any(len(v) != 4 or any(not 0 <= q <= 1 for q in v) for v in vs):
            raise RuntimeError('Equity leaf probabilities are invalid')
        for i, f in enumerate(fs):
            if f >= 0:
                if f >= 41 or not math.isfinite(ts[i]) or not i < ls[i] < n or not i < rs[i] < n:
                    raise RuntimeError('Equity tree branches are invalid')
            elif not 0 <= -f - 1 < len(vs):
                raise RuntimeError('Equity tree leaf index is invalid')
    roots = []
    for fs, ts, ls, rs, vs in _TREES[:TREE_LIMIT]:
        nodes = [None] * len(fs)
        for i in range(len(fs) - 1, -1, -1):
            f = fs[i]
            nodes[i] = (f, ts[i], nodes[ls[i]], nodes[rs[i]]) if f >= 0 else (-1, tuple(vs[-f - 1]))
        roots.append(nodes[0])
    # Drop indexed arrays and leaf lists: nested immutable tuples need fewer
    # pointer lookups per tree edge and permit garbage-collector untracking.
    _TREES = tuple(roots)
    # Immutable descendants can be untracked a level at a time by CPython's
    # collector. Complete this bounded startup work so a later collection does
    # not repeatedly scan the large constant model during a timed decision.
    for _ in range(24):
        if not any(gc.is_tracked(root) for root in _TREES):
            break
        gc.collect()
    _VALIDATED = True


def forced_chop(board):
    """True only when every legal holding must play the same five-card board."""
    if len(board) != 5:
        return False
    e = evaluate(board)
    cat, rest = divmod(e, 759375)
    high, rest = divmod(rest, 50625)
    kicker = rest // 3375
    if cat == 8 and high == 14:
        return True
    if cat == 7 and kicker == (13 if high == 14 else 14):
        return True
    if cat == 4 and high == 14:
        suits = [0] * 4
        for c in board:
            suits[c & 3] += 1
        return max(suits) <= 2
    return False


def hand_class(hole):
    a, b = hole
    h, l = max(a >> 2, b >> 2), min(a >> 2, b >> 2)
    return h * 13 + l if a % 4 == b % 4 else l * 13 + h


def preflop_percentile(hole):
    """Combo-weighted percentile, lower means a stronger starting hand."""
    if _PF is not None:
        return _PF['percentiles'][hand_class(hole)]
    a, b = sorted((c // 4 + 2 for c in hole), reverse=True)
    score = (a + b) / 28 + (.35 if a == b else 0) + (.05 if hole[0] % 4 == hole[1] % 4 else 0)
    return max(.001, min(.99, 1.15 - score))


def features(hole, board):
    """Suit-invariant 41-value feature vector shared with native training code."""
    hole = sorted(hole, key=lambda c: (c >> 2, sum((b & 3) == (c & 3) for b in board)), reverse=True)
    all_cards = list(board) + hole
    out = []
    for cards in (all_cards, board):
        e = evaluate(cards)
        out.append(e // 759375)
        e %= 759375
        for p in (50625, 3375, 225, 15, 1):
            out.append(e // p)
            e %= p
    br = sorted((c // 4 + 2 for c in board), reverse=True)
    out.extend(br + [0] * (5 - len(br)))
    bc, bs, ass = [0] * 13, [0] * 4, [0] * 4
    bm = am = 0
    for c in board:
        bc[c >> 2] += 1
        bs[c & 3] += 1
        bm |= 1 << (c >> 2)
    for c in all_cards:
        am |= 1 << (c >> 2)
        ass[c & 3] += 1
    h, l = hole[0] >> 2, hole[1] >> 2
    out.extend((h + 2, l + 2, int(hole[0] % 4 == hole[1] % 4)))
    mx = max(ass)
    out.extend((max(bs), bs[hole[0] & 3], bs[hole[1] & 3], mx))
    out.append(max((c // 4 + 2 for c in hole if ass[c & 3] == mx), default=0))
    best = besttop = boardbest = hits = 0
    hm = (1 << h) | (1 << l)
    for k in range(3, 13):
        m = 4111 if k == 3 else 31 << (k - 4)
        n, bn = (am & m).bit_count(), (bm & m).bit_count()
        if n >= best:
            best, besttop, hits = n, k + 2, (hm & m).bit_count()
        boardbest = max(boardbest, bn)
    out.extend((best, besttop, hits, boardbest, max(bc), bm.bit_count(), bc[h], bc[l], int(h + 2 > br[0]) + int(l + 2 > br[0]), h + 2 - br[0], l + 2 - br[0]))
    out.extend(sorted(bs, reverse=True))
    out.append(len(board))
    return out


def _interp(xs, ys, x):
    i = bisect.bisect_left(xs, x)
    if not i:
        return ys[0]
    if i >= len(xs):
        return ys[-1]
    t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] * (1 - t) + ys[i] * t


def estimate_all(hole, board):
    """Return equities vs 7%,20%,50%,100% preflop ranges."""
    if not board:
        if _PF is not None:
            ys = _PF['equities'][hand_class(hole)]
            return tuple(_interp(_PF['fractions'], ys, x) for x in _FRACS)
        q = .70 - .39 * preflop_percentile(hole)
        return (max(.12, q - .22), max(.18, q - .14), max(.23, q - .07), q)
    x = features(hole, board)
    # These outcomes are exact for every legal opponent holding.
    if x[0] == 8 and x[1] == 14:
        q = .5 if len(board) == 5 and x[6] == 8 and x[7] == 14 else 1.
        return (q, q, q, q)
    if len(board) == 5 and ((x[6] == 7 and x[8] == (13 if x[7] == 14 else 14)) or (x[6] == 4 and x[7] == 14 and x[20] <= 2)):
        return (.5, .5, .5, .5)
    if _TREES is not None:
        sums = [0., 0., 0., 0.]
        for node in _TREES:
            while node[0] >= 0:
                node = node[2] if x[node[0]] <= node[1] else node[3]
            v = node[1]
            sums[0] += v[0]
            sums[1] += v[1]
            sums[2] += v[2]
            sums[3] += v[3]
        n = len(_TREES)
        return tuple(max(0., min(1., s / n)) for s in sums)
    # Explicit developmental checkpoint only; final package contains the model.
    q = (.18, .58, .76, .87, .92, .96, .985, .998, 1.)[x[0]]
    return (max(0, q - .15), max(0, q - .09), max(0, q - .04), q)


def estimate(hole, board, range_fraction=1.0):
    if not board and _PF is not None:
        return _interp(_PF['fractions'], _PF['equities'][hand_class(hole)], range_fraction)
    return _interp(_FRACS, estimate_all(hole, board), range_fraction)


load()
if _TREES is not None:
    require_assets()
