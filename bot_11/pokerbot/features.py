"""Suit-invariant card features shared by training and the offline policy."""
from .evaluator import evaluate

FEATURE_NAMES = (
    'street_cards', 'hole_high', 'hole_low', 'pocket_pair', 'suited', 'gap',
    'made_category', 'kicker1', 'kicker2', 'kicker3', 'kicker4', 'kicker5',
    'board1', 'board2', 'board3', 'board4', 'board5',
    'board_distinct', 'board_pairs', 'board_trips', 'board_quads',
    'board_max_suit', 'board_second_suit', 'board_span',
    'high_matches', 'low_matches', 'overcards', 'underpair',
    'high_suit_count', 'low_suit_count', 'high_suit_above', 'low_suit_above',
    'best_flush_count', 'flush_hole_high', 'flush_hole_low', 'flush_above',
    'best_straight_count', 'best_straight_hole', 'board_straight_count',
    'straight_out_ranks', 'straight_nut_high', 'board_connect4',
    'hero_unique_high', 'hero_unique_low', 'board_highcard',
    'board_flush_high', 'board_flush_second', 'board_flush_third',
    'high_suit_board_high', 'low_suit_board_high',
    'own_pair_high', 'own_pair_low', 'set_rank', 'board_aces',
    'board_broadway', 'hole_broadway', 'board_low', 'wheel_cards',
)


def features(hole, board):
    """Encode exact made hand, ownership, blockers, draws and board texture."""
    a, b = sorted(hole, reverse=True)
    hi, lo = (a >> 2) + 2, (b >> 2) + 2
    hs, ls = a & 3, b & 3
    br = sorted(((c >> 2) + 2 for c in board), reverse=True)
    rc = [0] * 15
    sm = [0] * 4
    sc = [0] * 4
    bm = 0
    for c in board:
        r, s = (c >> 2) + 2, c & 3
        rc[r] += 1
        sm[s] |= 1 << r
        sc[s] += 1
        bm |= 1 << r
    if hi == lo and (sc[hs], sm[hs]) < (sc[ls], sm[ls]):
        hs, ls = ls, hs
    score = evaluate(tuple(hole) + tuple(board))
    hm = (1 << hi) | (1 << lo)
    full = bm | hm
    # Ace also occurs at rank 1 for wheel straight windows.
    full_w = full | ((full >> 13) & 2)
    board_w = bm | ((bm >> 13) & 2)
    best = boardbest = bestown = highstraight = 0
    outs = 0
    for top in range(5, 15):
        window = 31 << (top - 4)
        count = (full_w & window).bit_count()
        own = ((full_w & ~board_w) & window).bit_count()
        bc = (board_w & window).bit_count()
        boardbest = max(boardbest, bc)
        if count > best or (count == best and own > bestown):
            best, bestown = count, own
        if count >= 4:
            if count == 4:
                outs |= window & ~full_w
            highstraight = top
    # A and its wheel alias count as the same out.
    if outs & 2:
        outs = (outs & ~2) | (1 << 14)
    suit_order = sorted(range(4), key=lambda s: (sc[s] + (hs == s) + (ls == s), sc[s], sm[s], hi if hs == s else 0, lo if ls == s else 0), reverse=True)
    fs = suit_order[0]
    fh = max(hi if hs == fs else 0, lo if ls == fs else 0)
    fl = min(hi, lo) if hs == ls == fs else 0
    fa = sum(1 for r in range(fh + 1, 15) if not sm[fs] & (1 << r)) if fh else 0
    board_suit = max(range(4), key=lambda s: (sc[s], sm[s]))
    suited_ranks = sorted((r for r in range(2, 15) if sm[board_suit] & (1 << r)), reverse=True)
    suited_ranks += [0, 0, 0]
    counts = sorted(sc, reverse=True)
    ownpairs = sorted((r for r in set((hi, lo)) if rc[r] and (rc[r] + (hi == r) + (lo == r)) >= 2), reverse=True)
    ownpairs += [0, 0]
    setrank = hi if hi == lo and rc[hi] else 0
    result = [
        len(board), hi, lo, int(hi == lo), int(hs == ls), hi - lo,
        score >> 20,
        *[(score >> k) & 15 for k in (16, 12, 8, 4, 0)],
        *(br + [0] * (5 - len(br))),
        len(set(br)), sum(n == 2 for n in rc), sum(n == 3 for n in rc), sum(n == 4 for n in rc),
        counts[0], counts[1], br[0] - br[-1],
        rc[hi], rc[lo], int(hi > br[0]) + int(lo > br[0]), int(hi == lo and hi < br[0]),
        sc[hs], sc[ls],
        sum(1 for r in range(hi + 1, 15) if not sm[hs] & (1 << r)),
        sum(1 for r in range(lo + 1, 15) if not sm[ls] & (1 << r)),
        sc[fs] + (hs == fs) + (ls == fs), fh, fl, fa,
        best, bestown, boardbest, outs.bit_count(), highstraight, int(boardbest >= 4),
        int(not rc[hi]), int(not rc[lo]), br[0],
        *suited_ranks[:3], sm[hs].bit_length() - 1, sm[ls].bit_length() - 1,
        ownpairs[0], ownpairs[1], setrank, rc[14],
        sum(r >= 10 for r in br), int(hi >= 10) + int(lo >= 10), sum(r <= 5 for r in br),
        (full_w & 62).bit_count(),
    ]
    return result

