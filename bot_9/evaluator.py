"""Independent dependency-free Hold'em evaluation. Cards: rank*4+suit, 2=rank 0.

Ranks compare lexicographically via an integer; larger wins. Category occupies bits
20+, then five four-bit ranks (2..14). Callers supply 5..7 distinct valid cards.
"""
from itertools import combinations

CATEGORY_SHIFT = 20
CATEGORY_NAMES = ('high_card', 'pair', 'two_pair', 'trips', 'straight',
                  'flush', 'full_house', 'quads', 'straight_flush')


def _straight(mask):
    for high in range(12, 3, -1):
        if mask >> (high - 4) & 31 == 31:
            return high + 2
    return 5 if mask & 0x100F == 0x100F else 0


# Rank-mask lookup costs 8192 integers (~300KB) and makes evaluation predictable.
STRAIGHT = tuple(_straight(mask) for mask in range(8192))
TOP = tuple(tuple(r + 2 for r in range(12, -1, -1) if mask & (1 << r))
            for mask in range(8192))


def _pack(category, ranks):
    value = category << 20
    shift = 16
    for rank in ranks:
        value |= rank << shift
        shift -= 4
        if shift < 0:
            break
    return value


def evaluate(cards):
    """Evaluate exactly the best five cards from five, six, or seven cards."""
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    mask = 0
    for card in cards:
        rank = card >> 2
        bit = 1 << rank
        counts[rank] += 1
        suits[card & 3] |= bit
        mask |= bit
    flush_mask = 0
    for suit_mask in suits:
        if suit_mask.bit_count() >= 5:
            high = STRAIGHT[suit_mask]
            if high:
                return (8 << 20) | (high << 16)
            flush_mask = suit_mask
            break
    quads = trips = pair = pair2 = 0
    for rank in range(12, -1, -1):
        count = counts[rank]
        if count == 4:
            quads = rank + 2
        elif count == 3:
            if not trips:
                trips = rank + 2
            elif not pair:
                pair = rank + 2
            elif not pair2:
                pair2 = rank + 2
        elif count == 2:
            if not pair:
                pair = rank + 2
            elif not pair2:
                pair2 = rank + 2
    if quads:
        kicker = TOP[mask & ~(1 << (quads - 2))][0]
        return (7 << 20) | (quads << 16) | (kicker << 12)
    if trips and pair:
        return (6 << 20) | (trips << 16) | (pair << 12)
    if flush_mask:
        return _pack(5, TOP[flush_mask][:5])
    straight = STRAIGHT[mask]
    if straight:
        return (4 << 20) | (straight << 16)
    if trips:
        return _pack(3, (trips,) + TOP[mask & ~(1 << (trips - 2))][:2])
    if pair2:
        kicker = TOP[mask & ~(1 << (pair - 2)) & ~(1 << (pair2 - 2))][0]
        return (2 << 20) | (pair << 16) | (pair2 << 12) | (kicker << 8)
    if pair:
        return _pack(1, (pair,) + TOP[mask & ~(1 << (pair - 2))][:3])
    return _pack(0, TOP[mask][:5])


def reference_five(cards):
    """Slow independently structured five-card reference, intended for tests."""
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    groups = sorted(((ranks.count(rank), rank) for rank in set(ranks)), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    unique = sorted(set(ranks), reverse=True)
    straight = 0
    if len(unique) == 5:
        if unique[0] - unique[-1] == 4:
            straight = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight = 5
    if flush and straight:
        return _pack(8, [straight])
    if groups[0][0] == 4:
        return _pack(7, [groups[0][1], groups[1][1]])
    if [g[0] for g in groups] == [3, 2]:
        return _pack(6, [groups[0][1], groups[1][1]])
    if flush:
        return _pack(5, ranks)
    if straight:
        return _pack(4, [straight])
    if groups[0][0] == 3:
        return _pack(3, [g[1] for g in groups])
    if [g[0] for g in groups[:2]] == [2, 2]:
        return _pack(2, [g[1] for g in groups])
    if groups[0][0] == 2:
        return _pack(1, [g[1] for g in groups])
    return _pack(0, ranks)


def hand_features(hole, board):
    """Cheap, deterministic features; no simulation or hidden-card access.

    straight_draw gives number of distinct rank outs (4 or 8, rarely 12),
    excluding made straights; flush_draw requires a hole card in that suit.
    hole_used is a coarse ownership indicator, not an exact best-five count.
    """
    cards = tuple(hole) + tuple(board)
    value = evaluate(cards)
    category = value >> 20
    ranks = [c >> 2 for c in cards]
    branks = [c >> 2 for c in board]
    hranks = [c >> 2 for c in hole]
    counts = [ranks.count(rank) for rank in range(13)]
    board_counts = [branks.count(rank) for rank in range(13)]
    top_board = max(branks) if branks else -1
    pair_rank = max((rank for rank in hranks if counts[rank] >= 2), default=-1)
    relative_pair = 0
    if pair_rank >= 0:
        relative_pair = (4 if pair_rank > top_board else 3 if pair_rank == top_board
                         else 2 if sum(rank > pair_rank for rank in set(branks)) == 1 else 1)
    suits = [0] * 4
    bsuits = [0] * 4
    suit_masks = [0] * 4
    mask = bmask = 0
    for card in cards:
        suits[card & 3] += 1
        suit_masks[card & 3] |= 1 << (card >> 2)
        mask |= 1 << (card >> 2)
    for card in board:
        bsuits[card & 3] += 1
        bmask |= 1 << (card >> 2)
    flush_suit = next((s for s in range(4) if suits[s] == 4 and any(c & 3 == s for c in hole)), -1)
    nut_flush_draw = False
    if flush_suit >= 0:
        missing_top = next((r for r in range(12, -1, -1) if not any((c >> 2) == r and (c & 3) == flush_suit for c in board)), -1)
        nut_flush_draw = any((c >> 2) == missing_top and (c & 3) == flush_suit for c in hole)
    out_ranks = 0
    if not STRAIGHT[mask]:
        out_ranks = sum(1 for rank in range(13) if not (mask & (1 << rank)) and STRAIGHT[mask | (1 << rank)])
    board_straight = max((sum(bool(bmask & (1 << r)) for r in window)
                          for window in ([12, 0, 1, 2, 3],) + tuple(tuple(range(low, low + 5)) for low in range(9))), default=0)
    # Exact board-play check only on river; category equality alone is inadequate.
    board_rank = evaluate(board) if len(board) == 5 else 0
    hole_used = 0 if board_rank == value else 1
    if len(board) >= 3:
        # Pair ownership is more strategically useful than arbitrary tie selections.
        hole_used = max(hole_used, sum(counts[r] > board_counts[r] and counts[r] >= 2 for r in set(hranks)))
    return {
        'rank': value, 'category': category, 'pair_rank': pair_rank + 2 if pair_rank >= 0 else 0,
        'relative_pair': relative_pair, 'flush_draw': flush_suit >= 0 and len(board) < 5,
        'nut_flush_draw': nut_flush_draw and len(board) < 5,
        'straight_draw': out_ranks * 4 if len(board) < 5 else 0,
        'overcards': sum(rank > top_board for rank in hranks),
        'board_paired': max(board_counts, default=0) >= 2,
        'board_flush': max(bsuits, default=0), 'board_straight': board_straight,
        'board_rank': board_rank, 'plays_board': board_rank == value,
        'hole_used': hole_used,
    }
