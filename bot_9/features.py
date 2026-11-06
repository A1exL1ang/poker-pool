"""Suit-invariant, public-information equity features (also used during training)."""
from evaluator import evaluate, STRAIGHT

WINDOWS = (0x100F,) + tuple(31 << low for low in range(9))


def equity_features(hole, board):
    """Fixed schema. No opponent cards, random sampling, or action history."""
    hr = sorted((c >> 2 for c in hole), reverse=True)
    br = sorted((c >> 2 for c in board), reverse=True)
    bc = [0] * 13
    bs = [0] * 4
    hs = [0] * 4
    sm = [0] * 4
    bm = mask = 0
    for c in board:
        r, s = c >> 2, c & 3
        bc[r] += 1
        bs[s] += 1
        sm[s] |= 1 << r
        bm |= 1 << r
    for c in hole:
        s = c & 3
        hs[s] += 1
        sm[s] |= 1 << (c >> 2)
    mask = bm | (1 << hr[0]) | (1 << hr[1])
    rank = evaluate(tuple(hole) + tuple(board))
    board_rank = evaluate(board) if len(board) == 5 else 0
    x = [len(board), hr[0], hr[1], int(hr[0] == hr[1]),
         int(hole[0] & 3 == hole[1] & 3), hr[0] - hr[1],
         rank >> 20]
    x.extend((rank >> shift) & 15 for shift in (16, 12, 8, 4, 0))
    x.extend(br + [-1] * (5 - len(br)))
    x.extend(bc)
    x.extend((bc[hr[0]], bc[hr[1]], sum(r > br[0] for r in hr),
              sum(r > hr[0] for r in set(br)), sum(r > hr[1] for r in set(br)),
              max(bc), len(set(br)), board_rank >> 20,
              int(board_rank == rank)))
    # Canonical suits carry both card counts and the hero's flush blockers.
    suit_order = sorted(range(4), key=lambda s: (bs[s], hs[s], sm[s]), reverse=True)
    for s in suit_order:
        hero_suit = sorted((c >> 2 for c in hole if c & 3 == s), reverse=True)
        x.extend((bs[s], hs[s], hero_suit[0] if hero_suit else -1,
                  (sm[s].bit_length() - 1), STRAIGHT[sm[s]]))
    best_union = max((mask & w).bit_count() for w in WINDOWS)
    best_board = max((bm & w).bit_count() for w in WINDOWS)
    outs = sum(not (mask & (1 << r)) and bool(STRAIGHT[mask | (1 << r)]) for r in range(13))
    x.extend((best_union, best_board, outs, STRAIGHT[mask], STRAIGHT[bm]))
    # Each possible straight window: occupancy and hole contribution.
    for w in WINDOWS:
        x.extend(((bm & w).bit_count(), (mask & w).bit_count()))
    return x


N_FEATURES = len(equity_features((48, 45), (0, 5, 10)))
