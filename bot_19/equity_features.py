"""Cheap relational features derived solely from the stable 43-value input.

No cards, hidden information, new native labels, or online estimation are
required. Their purpose is to make board counterfeiting and straight rank
relationships easier for a compact tree model to represent.
"""
from poker_eval import FEATURE_NAMES

DERIVED_NAMES = (
    "primary_rank_gain", "secondary_rank_gain", "third_rank_gain",
    "straight_board_gain", "straight_nut_gap", "highest_board_pair",
    "lowest_board_pair", "board_trip_rank", "made_rank_vs_board_pair",
    "second_pair_vs_board_pair", "board_cards_above_kicker",
    "private_high_vs_board_high",
)
MODEL_FEATURE_NAMES = FEATURE_NAMES + DERIVED_NAMES


def derived_features(f):
    nb, cat = int(f[0]), int(f[6])
    ranks = f[12:12+nb]
    counts = {}
    mask = 0
    for rank in ranks:
        r = int(rank)
        counts[r] = counts.get(r, 0) + 1
        if 2 <= r <= 14:
            mask |= 1 << (r-2)
    pairs = [r for r,n in counts.items() if n >= 2]
    high_pair = max(pairs, default=0)
    low_pair = min(pairs, default=0)
    trip = max((r for r,n in counts.items() if n >= 3), default=0)
    nut_straight = 0
    if cat == 4:
        for i in range(9):
            if (mask & (31 << i)).bit_count() >= 3:
                nut_straight = i + 6
        if not nut_straight and (mask & 0x100f).bit_count() >= 3:
            nut_straight = 5
    kicker = f[9] if cat == 2 else f[8]
    return [f[7]-f[35], f[8]-f[36], f[9]-f[37],
            f[7]-f[33] if cat == 4 else 0,
            f[7]-nut_straight if cat == 4 else 0,
            high_pair, low_pair, trip,
            f[7]-high_pair if cat in (1,2,3,6) else 0,
            f[8]-low_pair if cat == 2 else 0,
            sum(r > kicker for r in ranks) if cat in (1,2,3,7) else 0,
            f[1]-f[12]]


def extend_features(f):
    return list(f) + derived_features(f)


def board_plays(f):
    """Whether hero's full best-five rank equals the five-card public board.

    Equality includes every kicker, especially the low kicker omitted from
    the three compact board-rank features. This is a ranking fact, not an
    equity estimate: a board-playing hero can tie or lose, never win outright.
    """
    if f[0] != 5 or f[6] != f[34]:
        return False
    cat = int(f[6])
    if cat in (0, 5):
        return f[7:12] == f[12:17]
    if cat == 1:
        pair = f[35]
        kickers = [r for r in f[12:17] if r != pair]
        return len(kickers) == 3 and f[7] == pair and tuple(f[8:11]) == tuple(kickers)
    if cat in (2, 3):
        return f[7:10] == f[35:38]
    if cat in (4, 8):
        return f[7] == f[35]
    if cat in (6, 7):
        return f[7:9] == f[35:37]
    return False
