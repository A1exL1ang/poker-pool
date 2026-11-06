"""Original bit-mask Hold'em evaluator and deterministic model features.

Cards are ``4 * (rank - 2) + suit``. Scores are lexicographically packed
``category, five tiebreak digits``; larger is better. Category 0..8 is high
card through straight flush. No third-party gameplay dependency is needed.
"""

from itertools import combinations


def _straight(mask):
    runs = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    if runs:
        return runs.bit_length() + 5
    return 5 if mask & 0x100F == 0x100F else 0


STRAIGHT = tuple(_straight(i) for i in range(8192))


def _top(mask, count):
    value = 0
    for index in range(count):
        if not mask:
            break
        bit = mask.bit_length() - 1
        value |= (bit + 2) << (16 - 4 * index)
        mask ^= 1 << bit
    return value


TOP5 = tuple(_top(i, 5) for i in range(8192))
TOP3 = tuple((_top(i, 3) >> 4) for i in range(8192))
TOP2 = tuple((_top(i, 2) >> 4) for i in range(8192))
WINDOWS = tuple(31 << i for i in range(9)) + (0x100F,)


def evaluate(cards):
    """Return best five-card rank from 5..7 distinct integer cards.

    For feature extraction only, 3 or 4 cards are also accepted with absent
    kickers below the lowest rank. Only 5..7 cards represent legal poker hand
    ranks. Input validation belongs at the API boundary.
    """
    mask = pairs = trips = quads = 0
    suits = [0, 0, 0, 0]
    for card in cards:
        bit = 1 << (card >> 2)
        quads |= trips & bit
        trips |= pairs & bit
        pairs |= mask & bit
        mask |= bit
        suits[card & 3] |= bit
    flush = 0
    for suit in suits:
        if suit.bit_count() >= 5:
            high = STRAIGHT[suit]
            if high:
                return 0x800000 | (high << 16)
            flush = TOP5[suit]
            break
    if quads:
        bit = quads.bit_length() - 1
        kicker = (mask ^ (1 << bit)).bit_length() + 1
        return 0x700000 | ((bit + 2) << 16) | (kicker << 12)
    if trips:
        bit = trips.bit_length() - 1
        pairmask = pairs & ~(1 << bit)
        if pairmask:
            return 0x600000 | ((bit + 2) << 16) | ((pairmask.bit_length() + 1) << 12)
    if flush:
        return 0x500000 | flush
    high = STRAIGHT[mask]
    if high:
        return 0x400000 | (high << 16)
    if trips:
        return 0x300000 | ((bit + 2) << 16) | TOP2[mask & ~(1 << bit)]
    if pairs:
        bit = pairs.bit_length() - 1
        rest = pairs & ~(1 << bit)
        if rest:
            second = rest.bit_length() - 1
            kicker = (mask & ~(1 << bit) & ~(1 << second)).bit_length() + 1
            return 0x200000 | ((bit + 2) << 16) | ((second + 2) << 12) | (kicker << 8)
        return 0x100000 | ((bit + 2) << 16) | TOP3[mask & ~(1 << bit)]
    return TOP5[mask]


FEATURE_NAMES = (
    "board_count", "hole_high", "hole_low", "pocket_pair", "suited", "gap",
    "category", "rank_1", "rank_2", "rank_3", "rank_4", "rank_5",
    "board_high", "board_2", "board_3", "board_4", "board_5",
    "board_unique", "board_max_multiplicity", "board_pair_ranks", "board_trip_ranks",
    "board_max_suit", "combined_max_suit", "flush_hole_count", "flush_hole_high",
    "straight_window", "board_straight_window", "straight_out_ranks",
    "overcards", "high_matches_board", "low_matches_board", "pocket_overcards",
    "flush_blocker_high", "board_straight_high", "board_category", "board_rank_1",
    "board_rank_2", "board_rank_3", "category_gain", "hole_window_contribution",
    "flush_board_count", "range_width", "aggression",
)


def features(hole, board, range_width=1.0, aggression=0.0):
    """Stable 43-feature contract shared exactly with the native generator."""
    r0 = (hole[0] >> 2) + 2
    r1 = (hole[1] >> 2) + 2
    hi, lo = (r0, r1) if r0 >= r1 else (r1, r0)
    ranks = sorted(((c >> 2) + 2 for c in board), reverse=True)
    bc = [0] * 15
    bs = [0] * 4
    hs = [0] * 4
    bm = hm = 0
    for c in board:
        bc[(c >> 2) + 2] += 1
        bs[c & 3] += 1
        bm |= 1 << (c >> 2)
    for c in hole:
        hs[c & 3] += 1
        hm |= 1 << (c >> 2)
    counts = [bs[i] + hs[i] for i in range(4)]
    fs = max(range(4), key=lambda i: (counts[i], hs[i], max(((c >> 2) + 2 for c in hole if c & 3 == i), default=0)))
    fh = max(((c >> 2) + 2 for c in hole if c & 3 == fs), default=0)
    mask = bm | hm
    sw = bsw = hc = 0
    outs = 0
    for window in WINDOWS:
        n = (mask & window).bit_count()
        sw = max(sw, n)
        bsw = max(bsw, (bm & window).bit_count())
        if n == 4:
            outs |= window & ~mask
        if n >= 4:
            hc = max(hc, (hm & window & ~bm).bit_count())
    rank = evaluate(list(hole) + list(board))
    brank = evaluate(board)
    category = rank >> 20
    bcategory = brank >> 20
    blockers = max(((c >> 2) + 2 for c in hole if bs[c & 3] >= 3), default=0)
    out = [len(board), hi, lo, int(hi == lo), int(hole[0] & 3 == hole[1] & 3), hi - lo,
           category, (rank >> 16) & 15, (rank >> 12) & 15, (rank >> 8) & 15,
           (rank >> 4) & 15, rank & 15]
    out.extend(ranks + [0] * (5 - len(board)))
    out.extend([bm.bit_count(), max(bc), sum(n >= 2 for n in bc), sum(n >= 3 for n in bc),
                max(bs), counts[fs], hs[fs], fh, sw, bsw, outs.bit_count(),
                int(hi > ranks[0]) + int(lo > ranks[0]), bc[hi], bc[lo],
                sum(r > hi for r in ranks) if hi == lo else 0, blockers,
                STRAIGHT[bm], bcategory, (brank >> 16) & 15, (brank >> 12) & 15,
                (brank >> 8) & 15, category - bcategory, hc, bs[fs],
                range_width, aggression])
    return out


def reference_evaluate(cards):
    """Independent slow sorting-based five-card enumeration for tests only."""
    def five(hand):
        ranks = sorted((c // 4 + 2 for c in hand), reverse=True)
        counts = {r: ranks.count(r) for r in set(ranks)}
        groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
        unique = sorted(counts, reverse=True)
        straight = unique[0] if len(unique) == 5 and unique[0] - unique[-1] == 4 else 0
        if unique == [14, 5, 4, 3, 2]:
            straight = 5
        flush = len({c % 4 for c in hand}) == 1
        if straight and flush:
            cat, digits = 8, [straight]
        elif groups[0][0] == 4:
            cat, digits = 7, [groups[0][1], groups[1][1]]
        elif [g[0] for g in groups] == [3, 2]:
            cat, digits = 6, [groups[0][1], groups[1][1]]
        elif flush:
            cat, digits = 5, ranks
        elif straight:
            cat, digits = 4, [straight]
        elif groups[0][0] == 3:
            cat, digits = 3, [groups[0][1]] + sorted((r for n, r in groups[1:]), reverse=True)
        elif [g[0] for g in groups[:2]] == [2, 2]:
            cat, digits = 2, sorted((groups[0][1], groups[1][1]), reverse=True) + [groups[2][1]]
        elif groups[0][0] == 2:
            cat, digits = 1, [groups[0][1]] + sorted((r for n, r in groups[1:]), reverse=True)
        else:
            cat, digits = 0, ranks
        return (cat << 20) | sum(r << (16 - 4 * i) for i, r in enumerate(digits))
    return max(five(h) for h in combinations(cards, 5))
