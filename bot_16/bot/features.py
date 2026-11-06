"""Semantic v3 equity features; coupled to the packaged v3 model."""
from bot.features_base import make_features as _base_features, hand_index
from bot.evaluator import TOP

VERSION = 3
FEATURE_COUNT = 84
WINDOWS = tuple((31 << i, i + 6) for i in range(9)) + ((4111, 5),)


def make_features(hole, board, opponents=1, range_fraction=1.0, pressure=0.0):
    f = _base_features(hole, board, opponents, range_fraction, pressure)
    cat, bcat = f[6], f[12]
    score = cat << 20
    bscore = bcat << 20
    for i in range(5):
        score |= f[7 + i] << (16 - 4 * i)
        bscore |= f[13 + i] << (16 - 4 * i)
    primary, secondary = f[7], f[8]
    bc = [0] * 15
    hc = [0] * 15
    suit_board = [0] * 4
    suit_hole = [0] * 4
    mask = 0
    for c in board:
        r = c // 4 + 2
        bc[r] += 1
        suit_board[c % 4] |= 1 << (r - 2)
        mask |= 1 << (r - 2)
    for c in hole:
        r = c // 4 + 2
        hc[r] += 1
        suit_hole[c % 4] |= 1 << (r - 2)
    possible_straights = 0
    max_straight = 0
    for window, high in WINDOWS:
        if (mask & window).bit_count() >= 3:
            possible_straights += 1
            if high > max_straight:
                max_straight = high
    max_sf = 0
    max_flush_score = 0
    max_higher_flush = 0
    for s in range(4):
        bm, hm = suit_board[s], suit_hole[s]
        if bm.bit_count() < 3:
            continue
        avail = 8191 ^ (bm | hm)
        for window, high in WINDOWS:
            missing = window & ~bm
            if missing.bit_count() <= 2 and not missing & ~avail:
                max_sf = max(max_sf, high)
        top_two = 0
        rest = avail
        for _ in range(2):
            if rest:
                bit = 1 << (rest.bit_length() - 1)
                top_two |= bit
                rest ^= bit
        max_flush_score = max(max_flush_score, TOP[bm | top_two])
        if hm:
            max_higher_flush = max(max_higher_flush, (avail >> hm.bit_length()).bit_count())
    board_ranks = sorted(set(c // 4 + 2 for c in board), reverse=True)
    max_quads = 0
    max_trips = 0
    for r in board_ranks:
        available = 4 - bc[r] - hc[r]
        if not max_quads and bc[r] >= 2 and 4 - bc[r] <= available:
            max_quads = r
        if not max_trips and bc[r] >= 1 and max(0, 3 - bc[r]) <= available:
            max_trips = r
    full_trip = full_pair = 0
    # The highest possible full house from this public board + two legal cards.
    for t in (board_ranks if max(bc) >= 2 else ()):
        nt = max(0, 3 - bc[t])
        if nt > 2 or nt > 4 - bc[t] - hc[t]:
            continue
        for p in (range(14, 1, -1) if nt == 0 else board_ranks):
            if p == t:
                continue
            np = max(0, 2 - bc[p])
            if nt + np <= 2 and np <= 4 - bc[p] - hc[p]:
                full_trip, full_pair = t, p
                break
        if full_trip:
            break
    threat_category = 8 if max_sf else 7 if max_quads else 6 if full_trip else 5 if max_flush_score else 4 if max_straight else 3 if max_trips else 2
    board_pair_high = max((r for r in range(2, 15) if bc[r] >= 2), default=0)
    board_trip_high = max((r for r in range(2, 15) if bc[r] >= 3), default=0)
    extras = [
        int(score > bscore), cat - bcat, primary - f[13], secondary - f[14],
        int(primary > f[13]), int(secondary > f[14]),
        board_pair_high, board_trip_high,
        primary - board_pair_high if cat in (3, 6, 7) else 0,
        max_straight, possible_straights,
        primary - max_straight if cat in (4, 8) else 0,
        max_sf, max_quads, max_trips, full_trip, full_pair,
        threat_category, cat - threat_category,
        primary - full_trip if cat == 6 else 0,
        secondary - full_pair if cat == 6 and primary == full_trip else 0,
        primary - max_quads if cat == 7 else 0,
        primary - max_trips if cat == 3 else 0,
        max_higher_flush,
        int((score & ((1 << 20) - 1)) >= max_flush_score) if cat == 5 else 0,
        primary - ((max_flush_score >> 16) & 15) if cat == 5 else 0,
    ]
    return f[:-3] + extras + f[-3:]
