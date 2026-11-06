"""Deterministic suit-invariant features for the local equity surrogate."""
from .evaluator import evaluate_partial, STRAIGHT
from .preflop import hand_score, preflop_percentile


WINDOWS = tuple((31 << i) for i in range(9)) + (0x100F,)
FEATURE_VERSION = "v2-pair-suit-canonical"


def features(hole, board):
    h0, h1 = hole
    if h0 >> 2 < h1 >> 2:
        h0, h1 = h1, h0
    hr0, hr1 = h0 // 4 + 2, h1 // 4 + 2
    bc = [0] * 13
    bs = [0] * 4
    bsm = [0] * 4
    bm = 0
    for c in board:
        r, s = c >> 2, c & 3
        bc[r] += 1
        bs[s] += 1
        bsm[s] |= 1 << r
        bm |= 1 << r
    if hr0 == hr1 and (bs[h0 & 3], bsm[h0 & 3]) < (bs[h1 & 3], bsm[h1 & 3]):
        h0, h1 = h1, h0
    mask = bm | (1 << (hr0 - 2)) | (1 << (hr1 - 2))
    hvalue = evaluate_partial((*hole, *board))
    bvalue = evaluate_partial(board)
    out = [len(board), hr0, hr1, int((h0 & 3) == (h1 & 3)), int(hr0 == hr1),
           hand_score(hole), preflop_percentile(hole)]
    out.extend((hvalue >> shift) & 15 for shift in (20, 16, 12, 8, 4, 0))
    out.extend((bvalue >> shift) & 15 for shift in (20, 16, 12, 8, 4, 0))
    branks = sorted((c // 4 + 2 for c in board), reverse=True)
    out.extend(branks + [0] * (5 - len(branks)))
    out.extend((bc[hr0 - 2], bc[hr1 - 2],
                sum(r > hr0 for r in branks), sum(r > hr1 for r in branks),
                sum(n >= 2 for n in bc), sum(n >= 3 for n in bc),
                max(bs), bs[h0 & 3], bs[h1 & 3]))
    alls = bs.copy()
    alls[h0 & 3] += 1
    alls[h1 & 3] += 1
    out.extend((max(alls), len([x for x in bs if x]),
                sum(n == 1 for n in bc), sum(n == 1 for n in alls)))
    for c in (h0, h1):
        r, s = c >> 2, c & 3
        # Flush rank blockers: how many higher suited ranks are absent from board.
        out.extend((sum(not (bsm[s] & (1 << x)) for x in range(r + 1, 13)),
                    (bsm[s] >> (r + 1)).bit_count()))
    wcounts = [(mask & w).bit_count() for w in WINDOWS]
    bwcounts = [(bm & w).bit_count() for w in WINDOWS]
    maxw = max(wcounts)
    maxbw = max(bwcounts)
    outs = 0
    if not STRAIGHT[mask]:
        for r in range(13):
            if not (mask & (1 << r)) and STRAIGHT[mask | (1 << r)]:
                outs += 1
    out.extend((maxw, maxbw, wcounts.count(4), wcounts.count(3), outs,
                max((i + 6 if i < 9 else 5 for i, n in enumerate(wcounts) if n == maxw), default=0),
                mask.bit_count(), bm.bit_count()))
    out.extend(bc)
    # Hole rank indicators let trees learn blocker effects and board interactions.
    out.extend(int(hr0 == r) + int(hr1 == r) for r in range(2, 15))
    return out


FEATURE_COUNT = len(features((48, 45), (0, 5, 10)))
