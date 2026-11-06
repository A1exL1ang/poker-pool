"""Card encoding, exact best-five evaluator, and suit-invariant model features.

Cards are rank * 4 + suit, ranks 2..A mapped to 0..12, suits c,d,h,s.
No third-party evaluator or strategy code is used.
"""
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD_IDS = {r + s: i * 4 + j for i, r in enumerate(RANKS) for j, s in enumerate(SUITS)}
STRAIGHT = [0] * 8192
TOP5 = [0] * 8192
TOP3 = [0] * 8192
TOP2 = [0] * 8192
TOP1 = [0] * 8192
WINDOWS = [0x100F] + [31 << i for i in range(9)]
for _m in range(8192):
    _r = [r + 2 for r in range(12, -1, -1) if _m & (1 << r)]
    for _size, _table in ((1, TOP1), (2, TOP2), (3, TOP3), (5, TOP5)):
        _v = 0
        for _k in range(_size):
            _v = (_v << 4) | (_r[_k] if _k < len(_r) else 0)
        _table[_m] = _v
    if _m & 0x100F == 0x100F:
        STRAIGHT[_m] = 5
    for _i in range(9):
        if (_m >> _i) & 31 == 31:
            STRAIGHT[_m] = _i + 6
del _m, _r, _size, _table, _v, _k, _i


def card_id(card):
    if isinstance(card, int) and not isinstance(card, bool) and 0 <= card < 52:
        return card
    if isinstance(card, str):
        key = card.strip().replace("10", "T")
        if len(key) == 2:
            key = key[0].upper() + key[1].lower()
            if key in CARD_IDS:
                return CARD_IDS[key]
    raise ValueError("card must be an integer 0..51 or a rank/suit string such as As")


def card_name(card):
    return RANKS[card >> 2] + SUITS[card & 3]


def hand_class(a, b):
    r, s = a >> 2, b >> 2
    if r < s:
        r, s = s, r
    return r * 13 + s if (a & 3) == (b & 3) or r == s else s * 13 + r


def evaluate(cards):
    """Exact 5-7-card strength, larger is better; also useful on 2-4 cards.

    Category in bits >=20; five rank nibbles follow, all 2..14.
    """
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        bit = 1 << r
        mask |= bit
        suits[c & 3] |= bit
    for sm in suits:
        if sm.bit_count() >= 5:
            sh = STRAIGHT[sm]
            if sh:
                return (8 << 20) | (sh << 16)
            flush = (5 << 20) | TOP5[sm]
            break
    else:
        flush = 0
    quads = trips = pair1 = pair2 = 0
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            quads = r + 2
        elif n >= 3:
            if not trips:
                trips = r + 2
            elif not pair1:
                pair1 = r + 2
            elif not pair2:
                pair2 = r + 2
        elif n == 2:
            if not pair1:
                pair1 = r + 2
            elif not pair2:
                pair2 = r + 2
    if quads:
        return (7 << 20) | (quads << 16) | (TOP1[mask & ~(1 << (quads - 2))] << 12)
    if trips and pair1:
        return (6 << 20) | (trips << 16) | (pair1 << 12)
    if flush:
        return flush
    sh = STRAIGHT[mask]
    if sh:
        return (4 << 20) | (sh << 16)
    if trips:
        return (3 << 20) | (trips << 16) | (TOP2[mask & ~(1 << (trips - 2))] << 8)
    if pair2:
        pm = (1 << (pair1 - 2)) | (1 << (pair2 - 2))
        return (2 << 20) | (pair1 << 16) | (pair2 << 12) | (TOP1[mask & ~pm] << 8)
    if pair1:
        return (1 << 20) | (pair1 << 16) | (TOP3[mask & ~(1 << (pair1 - 2))] << 4)
    return TOP5[mask]


FEATURE_NAMES = (
    "board_count", "hole_high", "hole_low", "suited", "pocket_pair",
    "category", "kicker1", "kicker2", "kicker3", "kicker4", "kicker5",
    "board_category", "board_kicker1", "board_kicker2", "board_kicker3",
    "board_kicker4", "board_kicker5", "board_rank1", "board_rank2", "board_rank3",
    "board_rank4", "board_rank5", "board_unique", "board_max_duplicates",
    "board_max_suit", "high_hole_suit_count", "low_hole_suit_count",
    "all_max_suit", "max_suit_hole_count", "flush_hole_high", "flush_hole_low",
    "straight_count", "board_straight_count", "straight_out_ranks",
    "high_hole_board_matches", "low_hole_board_matches", "overcards",
    "board_over_pocket", "board_pair_count", "best_straight_window_high",
)


def features(hole, board):
    """Feature contract shared with tools/offline.cpp (40 integer features)."""
    a, b = sorted(hole, key=lambda c: (c >> 2, c & 3), reverse=True)
    hi, lo = (a >> 2) + 2, (b >> 2) + 2
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    bmask = 0
    for c in board:
        r = c >> 2
        counts[r] += 1
        suits[c & 3] += 1
        bmask |= 1 << r
    if hi == lo and suits[a & 3] < suits[b & 3]:
        a, b = b, a
    ranks = sorted(((c >> 2) + 2 for c in board), reverse=True)
    top = ranks[0] if ranks else 0
    ranks += [0] * (5 - len(ranks))
    all_suits = suits[:]
    all_suits[a & 3] += 1
    all_suits[b & 3] += 1
    best_suit = max(range(4), key=lambda s: (all_suits[s], int((a & 3) == s), int((b & 3) == s)))
    fhi = hi if (a & 3) == best_suit else 0
    flo = lo if (b & 3) == best_suit else 0
    if fhi < flo:
        fhi, flo = flo, fhi
    mask = bmask | (1 << (hi - 2)) | (1 << (lo - 2))
    sc, bc, wh = 0, 0, 0
    for i, w in enumerate(WINDOWS):
        c = (mask & w).bit_count()
        if c >= sc:
            sc, wh = c, i + 5
        bc = max(bc, (bmask & w).bit_count())
    outs = sum(bool(STRAIGHT[mask | (1 << r)]) for r in range(13) if not mask & (1 << r))
    v, bv = evaluate((*hole, *board)), evaluate(board)
    return [
        len(board), hi, lo, int((a & 3) == (b & 3)), int(hi == lo),
        v >> 20, (v >> 16) & 15, (v >> 12) & 15, (v >> 8) & 15, (v >> 4) & 15, v & 15,
        bv >> 20, (bv >> 16) & 15, (bv >> 12) & 15, (bv >> 8) & 15, (bv >> 4) & 15, bv & 15,
        *ranks, bmask.bit_count(), max(counts), max(suits),
        suits[a & 3], suits[b & 3], max(all_suits), int((a & 3) == best_suit) + int((b & 3) == best_suit), fhi, flo,
        sc, bc, outs, counts[hi - 2], counts[lo - 2], int(hi > top) + int(lo > top),
        sum(c > hi for c in ranks) if hi == lo else 0, sum(c >= 2 for c in counts), wh,
    ]


def reference_evaluate(cards):
    """Slow independent five-card oracle, only for validation/simulation checks."""
    def five(cs):
        ranks = sorted(((c >> 2) + 2 for c in cs), reverse=True)
        groups = sorted(((ranks.count(r), r) for r in set(ranks)), reverse=True)
        flush = len({c & 3 for c in cs}) == 1
        unique = sorted(set(ranks), reverse=True)
        straight = 5 if unique == [14, 5, 4, 3, 2] else (unique[0] if len(unique) == 5 and unique[0] - unique[-1] == 4 else 0)
        if straight and flush:
            cat, ks = 8, [straight]
        elif groups[0][0] == 4:
            cat, ks = 7, [groups[0][1], groups[1][1]]
        elif [g[0] for g in groups] == [3, 2]:
            cat, ks = 6, [g[1] for g in groups]
        elif flush:
            cat, ks = 5, ranks
        elif straight:
            cat, ks = 4, [straight]
        elif groups[0][0] == 3:
            cat, ks = 3, [groups[0][1]] + sorted([g[1] for g in groups[1:]], reverse=True)
        elif [g[0] for g in groups[:2]] == [2, 2]:
            cat, ks = 2, sorted([groups[0][1], groups[1][1]], reverse=True) + [groups[2][1]]
        elif groups[0][0] == 2:
            cat, ks = 1, [groups[0][1]] + sorted([g[1] for g in groups[1:]], reverse=True)
        else:
            cat, ks = 0, ranks
        return (cat << 20) | sum(k << (16 - i * 4) for i, k in enumerate(ks))
    return max(five(cs) for cs in combinations(cards, 5))
