"""Deterministic, card-only features for the offline equity model."""
from cards import evaluate

_WINDOWS = (0x100F,) + tuple(31 << low for low in range(9))


def _straight_summary(mask):
    counts = [(mask & window).bit_count() for window in _WINDOWS]
    best = max(counts)
    return best, max(i+5 for i,n in enumerate(counts) if n==best), counts.count(4)


# Fixed rank-mask combinatorics, independent of training data and opponents.
_STRAIGHT = tuple(_straight_summary(mask) for mask in range(8192))


def features(hole, board):
    hr = sorted((c // 4 + 2 for c in hole), reverse=True)
    br = [c // 4 + 2 for c in board]
    hc = [0] * 13
    bc = [0] * 13
    bs = [0] * 4
    for c in hole:
        hc[c // 4] += 1
    for c in board:
        bc[c // 4] += 1
        bs[c % 4] += 1
    val = evaluate(list(hole) + list(board))
    # Shared evaluator format: category << 20, then five descending nibbles.
    out = [len(board), hr[0], hr[1], int(hole[0] % 4 == hole[1] % 4)]
    out += hc + bc + sorted(bs, reverse=True)
    out += [bs[hole[0] % 4], bs[hole[1] % 4], val >> 20]
    out += [(val >> shift) & 15 for shift in (16, 12, 8, 4, 0)]
    # Counts and high ends of four/five-card straight windows; hole use matters.
    bm = 0
    for c in board:
        bm |= 1 << (c//4)
    out += _STRAIGHT[bm | (1 << (hole[0]//4)) | (1 << (hole[1]//4))]
    out += _STRAIGHT[bm]
    for c in hole:
        rank = c // 4 + 2
        out += [sum(r > rank for r in br), bc[rank - 2]]
    out += [len(set(br)), max(br), min(br), max(bc)]
    return out
