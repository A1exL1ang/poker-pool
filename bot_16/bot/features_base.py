"""Deterministic, suit-invariant features for offline learned showdown equity.

No hidden cards or random sampling are used at decision time. Feature ordering is
part of the saved model version and must be changed together with retraining.
"""
from bot.evaluator import evaluate

VERSION = 2
WINDOWS = tuple([31 << i for i in range(9)] + [4111])


def hand_index(hole):
    a, b = sorted((hole[0] // 4, hole[1] // 4), reverse=True)
    # 13 diagonal pairs, then 78 suited, then 78 offsuit combinations.
    if a == b:
        return a
    offset = a * (a - 1) // 2 + b
    return 13 + offset + (0 if hole[0] % 4 == hole[1] % 4 else 78)


def make_features(hole, board, opponents=1, range_fraction=1.0, pressure=0.0):
    ranks = [c // 4 + 2 for c in hole]
    h1, h2 = sorted(ranks, reverse=True)
    br = sorted((c // 4 + 2 for c in board), reverse=True)
    bc = [0] * 15
    suit_masks = [0] * 4
    board_suit_masks = [0] * 4
    bs = [0] * 4
    als = [0] * 4
    mask = 0
    bmask = 0
    for c in board:
        r, s = c // 4 + 2, c % 4
        bc[r] += 1
        suit_masks[s] |= 1 << (r - 2)
        board_suit_masks[s] |= 1 << (r - 2)
        bs[s] += 1
        als[s] += 1
        mask |= 1 << (r - 2)
        bmask |= 1 << (r - 2)
    for c in hole:
        r, s = c // 4 + 2, c % 4
        suit_masks[s] |= 1 << (r - 2)
        als[s] += 1
        mask |= 1 << (r - 2)
    score = evaluate(list(hole) + list(board))
    cat = score >> 20
    kickers = [(score >> shift) & 15 for shift in (16, 12, 8, 4, 0)]
    bscore = evaluate(board)
    bcat = bscore >> 20
    bkick = [(bscore >> shift) & 15 for shift in (16, 12, 8, 4, 0)]
    top = br[0]
    # Each five-rank straight window (including A2345) contributes to draw texture.
    best = 0
    bbest = 0
    outs = 0
    bouts = 0
    three = 0
    for w in WINDOWS:
        n = (mask & w).bit_count()
        bn = (bmask & w).bit_count()
        if n > best:
            best = n
        if bn > bbest:
            bbest = bn
        if n == 4:
            outs |= w & ~mask
        if bn == 4:
            bouts |= w & ~bmask
        if n == 3:
            three += 1
    # Flush potential and blockers; null suit identity to preserve symmetry.
    flush_suit = max(range(4), key=lambda s: (als[s], bs[s], suit_masks[s], board_suit_masks[s]))
    hole_flush = sorted((c // 4 + 2 for c in hole if c % 4 == flush_suit), reverse=True)
    board_flush = sorted((c // 4 + 2 for c in board if c % 4 == flush_suit), reverse=True)
    paired = sorted((r for r in range(2, 15) if bc[r] >= 2), reverse=True)
    matched = sorted((r for r in ranks if bc[r]), reverse=True)
    f = [
        len(board), h1, h2, int(h1 == h2), int(hole[0] % 4 == hole[1] % 4), h1 - h2,
        cat, *kickers, bcat, *bkick,
        *(br + [0] * (5 - len(br))),
        len(set(br)), max(bc), max(bs), max(als),
        sum(bc[r] >= 2 for r in range(2, 15)),
        sum(bc[r] >= 3 for r in range(2, 15)),
        paired[0] if paired else 0,
        sum(r > top for r in ranks),
        sum(r > top for r in ranks) * int(h1 == h2),
        bc[h1], bc[h2],
        matched[0] if matched else 0,
        sum(r > h1 for r in br), sum(r > h2 for r in br),
        best, bbest, outs.bit_count(), bouts.bit_count(), three,
        als[flush_suit], bs[flush_suit], len(hole_flush),
        hole_flush[0] if hole_flush else 0,
        hole_flush[1] if len(hole_flush) > 1 else 0,
        board_flush[0] if board_flush else 0,
        sum(r > (hole_flush[0] if hole_flush else 0) for r in board_flush),
        sum(1 for c in hole if bs[c % 4] >= 3),
        int(any(c // 4 == 12 and bs[c % 4] >= 2 for c in hole)),
        (mask & 7936).bit_count(), (bmask & 7936).bit_count(),
        int(cat > bcat), int(h1 == 14),
        opponents, range_fraction, pressure,
    ]
    return f
