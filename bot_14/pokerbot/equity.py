"""Fast suit-invariant postflop features for offline equity modelling.

This module does not claim a solver-quality range or run Monte Carlo during
gameplay. :func:`feature_vector` is the stable feature interface used by the
offline trainer; :func:`hand_features` is its readable diagnostic equivalent.
All feature construction is independent original code.
"""

from __future__ import annotations

from .evaluator import evaluate, _STRAIGHT

FEATURE_NAMES = (
    "board_length", "category", "made_rank_1", "made_rank_2", "made_rank_3", "made_rank_4", "made_rank_5",
    "hole_high", "hole_low", "hole_pair", "hole_suited",
    "board_rank_1", "board_rank_2", "board_rank_3", "board_rank_4", "board_rank_5",
    "board_unique_ranks", "board_max_multiplicity", "board_pairs", "board_trips",
    "board_max_suit", "total_max_suit", "flush_hole_count", "flush_hole_high", "flush_hole_low", "flush_nut_gap",
    "straight_window_count", "board_straight_window_count", "straight_out_ranks", "straight_window_high",
    "hole_overcards", "hole_high_board_hits", "hole_low_board_hits",
    "board_category", "board_made_rank_1", "board_made_rank_2", "board_made_rank_3", "board_made_rank_4", "board_made_rank_5",
    "board_suit_of_high_hole", "board_suit_of_low_hole", "board_gap", "hole_gap", "board_ace",
)

_WINDOWS = tuple((0x1F << low, low + 6) for low in range(9)) + ((0x100F, 5),)
_STRAIGHT_WINDOW_COUNT = [0] * 8192
_STRAIGHT_WINDOW_HIGH = [0] * 8192
_STRAIGHT_OUTS = [0] * 8192
for _mask in range(8192):
    _best = _high = 0
    _outs = 0
    for _window, _top in _WINDOWS:
        _count = (_mask & _window).bit_count()
        if _count > _best or _count == _best and _top > _high:
            _best, _high = _count, _top
        if _count == 4:
            _outs |= _window & ~_mask
    _STRAIGHT_WINDOW_COUNT[_mask] = _best
    _STRAIGHT_WINDOW_HIGH[_mask] = _high
    _STRAIGHT_OUTS[_mask] = 0 if _STRAIGHT[_mask] else _outs.bit_count()
del _mask, _best, _high, _outs, _window, _top, _count


def feature_vector(hole, board):
    """Return the 44 integer features named by :data:`FEATURE_NAMES`.

    ``hole`` contains two distinct card IDs and ``board`` three to five IDs.
    Like the evaluator, this timed primitive assumes valid unique input.
    Features retain exact made-hand kickers, hole involvement, blockers,
    straight/flush potential and texture. No opponent-specific information is
    included; the trainer conditions equity on its separately supplied range.
    """
    h0, h1 = hole
    if h0 // 4 < h1 // 4:
        h0, h1 = h1, h0
    hi, lo = h0 // 4 + 2, h1 // 4 + 2
    hs0, hs1 = h0 & 3, h1 & 3
    board_ranks = [0] * 13
    board_suits = [0] * 4
    board_suit_masks = [0] * 4
    board_mask = 0
    ranks_sorted = []
    for card in board:
        r, s = card >> 2, card & 3
        board_ranks[r] += 1
        board_suits[s] += 1
        board_suit_masks[s] |= 1 << r
        board_mask |= 1 << r
        ranks_sorted.append(r + 2)
    ranks_sorted.sort(reverse=True)
    padded_ranks = ranks_sorted + [0] * (5 - len(board))
    total_suits = board_suits[:]
    total_suits[hs0] += 1
    total_suits[hs1] += 1
    # Pick the most relevant suit; tie-breakers retain our strongest blocker.
    suit = max(range(4), key=lambda s: (total_suits[s], (hi if hs0 == s else lo if hs1 == s else 0), board_suit_masks[s]))
    flush_hi = hi if hs0 == suit else lo if hs1 == suit else 0
    flush_lo = lo if hs0 == suit and hs1 == suit else 0
    flush_count = (hs0 == suit) + (hs1 == suit)
    nut_gap = ((~board_suit_masks[suit] & 8191) >> (flush_hi - 1)).bit_count() if flush_hi else 13
    score = evaluate(tuple(hole) + tuple(board))
    board_score = evaluate(board) if len(board) == 5 else 0
    mask = board_mask | (1 << (hi - 2)) | (1 << (lo - 2))
    return (
        len(board), score >> 20, (score >> 16) & 15, (score >> 12) & 15, (score >> 8) & 15, (score >> 4) & 15, score & 15,
        hi, lo, int(hi == lo), int(hs0 == hs1),
        *padded_ranks,
        board_mask.bit_count(), max(board_ranks), sum(c == 2 for c in board_ranks), sum(c == 3 for c in board_ranks),
        max(board_suits), total_suits[suit], flush_count, flush_hi, flush_lo, nut_gap,
        _STRAIGHT_WINDOW_COUNT[mask], _STRAIGHT_WINDOW_COUNT[board_mask], _STRAIGHT_OUTS[mask], _STRAIGHT_WINDOW_HIGH[mask],
        int(hi > ranks_sorted[0]) + int(lo > ranks_sorted[0]), board_ranks[hi - 2], board_ranks[lo - 2],
        board_score >> 20, (board_score >> 16) & 15, (board_score >> 12) & 15, (board_score >> 8) & 15, (board_score >> 4) & 15, board_score & 15,
        max(board_suits[hs0], board_suits[hs1]) if hi == lo else board_suits[hs0],
        min(board_suits[hs0], board_suits[hs1]) if hi == lo else board_suits[hs1],
        ranks_sorted[0] - ranks_sorted[-1], hi - lo, int(bool(board_mask & 4096)),
    )


def hand_features(hole, board):
    """Readable mapping of the exact features used by the offline trainer."""
    return dict(zip(FEATURE_NAMES, feature_vector(hole, board)))


def has_absolute_nuts(hole, board):
    """Conservative constant-time detector for a subset of absolute nuts.

    True is a proof that no opponent can beat the hand (ties are possible).
    False is inconclusive. This intentionally does not enumerate 990 river
    hands inside a one-millisecond gameplay budget.
    """
    if len(board) < 3:
        return False
    cards = tuple(hole) + tuple(board)
    score = evaluate(cards)
    category = score >> 20
    high = score >> 16 & 15
    if category == 8 and high == 14:
        return True
    if len(board) != 5:
        return False
    # A straight flush needs at least three board cards of its suit.
    if category in (4, 7):
        suit_counts = [0] * 4
        rank_counts = [0] * 13
        for card in board:
            suit_counts[card & 3] += 1
            rank_counts[card >> 2] += 1
        if category == 4:
            return high == 14 and max(rank_counts) == 1 and max(suit_counts) < 3
        # Board quads need the best legal kicker: king when the quads are
        # aces, ace otherwise. It may already be provided by the board.
        if max(rank_counts) == 4:
            return (score >> 12 & 15) == (13 if high == 14 else 14) and max(suit_counts) < 3
        higher_board_trips = any(count >= 3 and rank + 2 > high for rank, count in enumerate(rank_counts))
        return max(suit_counts) < 3 and not higher_board_trips
    return False
