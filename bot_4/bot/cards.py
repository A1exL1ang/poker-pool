"""Original card evaluator and shared equity-model features. No dependencies.

Card integers: rank * 4 + suit, ranks 0..12 = 2..A, suits c,d,h,s.
Scores increase with hand strength. Works with 3..7 cards (partial board
scores below five cards are features, not complete poker hands).
"""
RANKS = "23456789TJQKA"
SUITS = "cdhs"
STRAIGHT = [0] * 8192
for _mask in range(8192):
    for _high in range(12, 3, -1):
        if (_mask >> (_high - 4)) & 31 == 31:
            STRAIGHT[_mask] = _high + 2
            break
    if not STRAIGHT[_mask] and _mask & 4111 == 4111:
        STRAIGHT[_mask] = 5
WINDOWS = [(31 << shift, shift + 6) for shift in range(9)] + [(4111, 5)]
COVER = [0] * 8192
FOUR_HIGH = [0] * 8192
OUT_COUNT = [0] * 8192
for _mask in range(8192):
    _outs = 0
    for _window, _high in WINDOWS:
        _n = (_mask & _window).bit_count()
        if _n > COVER[_mask]:
            COVER[_mask] = _n
        if _n == 4:
            FOUR_HIGH[_mask] = max(FOUR_HIGH[_mask], _high)
            _outs |= _window & ~_mask
    OUT_COUNT[_mask] = _outs.bit_count()


def card(value):
    if isinstance(value, int):
        if 0 <= value < 52:
            return value
        raise ValueError("card integer outside 0..51")
    value = value.strip()
    if value.startswith("10"):
        value = "T" + value[2:]
    if len(value) != 2:
        raise ValueError("cards use strings such as As, Td, 2c")
    return RANKS.index(value[0].upper()) * 4 + SUITS.index(value[1].lower())


def card_name(value):
    return RANKS[value >> 2] + SUITS[value & 3]


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
    counts = [0] * 15
    suits = [0] * 4
    mask = 0
    for c in cards:
        rank = (c >> 2) + 2
        counts[rank] += 1
        suits[c & 3] |= 1 << (rank - 2)
        mask |= 1 << (rank - 2)
    flush = 0
    for sm in suits:
        if sm.bit_count() >= 5:
            if STRAIGHT[sm]:
                return _pack(8, (STRAIGHT[sm],))
            flush = sm
            break
    fours = trips = 0
    pairs = []
    ranks = []
    for r in range(14, 1, -1):
        n = counts[r]
        if n:
            ranks.append(r)
            if n == 4:
                fours = r
            elif n == 3:
                if not trips:
                    trips = r
                else:
                    pairs.append(r)
            elif n == 2:
                pairs.append(r)
    if fours:
        return _pack(7, (fours, next((r for r in ranks if r != fours), 0)))
    if trips and pairs:
        return _pack(6, (trips, max(pairs)))
    if flush:
        return _pack(5, [r for r in range(14, 1, -1) if flush & (1 << (r - 2))][:5])
    if STRAIGHT[mask]:
        return _pack(4, (STRAIGHT[mask],))
    if trips:
        return _pack(3, [trips] + [r for r in ranks if r != trips][:2])
    if len(pairs) >= 2:
        return _pack(2, pairs[:2] + [r for r in ranks if r not in pairs[:2]][:1])
    if pairs:
        return _pack(1, pairs[:1] + [r for r in ranks if r != pairs[0]][:3])
    return _pack(0, ranks[:5])


def preflop_index(hole):
    a, b = hole
    hi, lo = sorted((a >> 2, b >> 2), reverse=True)
    return hi * 13 + lo if (a & 3) == (b & 3) or hi == lo else lo * 13 + hi


def features(hole, board):
    """43 cheap features; identical ordering in tools/equity.cpp."""
    a, b = hole
    hi, lo = sorted(((a >> 2) + 2, (b >> 2) + 2), reverse=True)
    score = evaluate((*hole, *board))
    bscore = evaluate(board)
    counts = [0] * 15
    bsuits = [0] * 4
    bmasks = [0] * 4
    mask = 0
    for c in board:
        r = (c >> 2) + 2
        counts[r] += 1
        bsuits[c & 3] += 1
        bmasks[c & 3] |= 1 << (r - 2)
        mask |= 1 << (r - 2)
    amask = mask | (1 << (hi - 2)) | (1 << (lo - 2))
    asuits = bsuits.copy()
    asuits[a & 3] += 1
    asuits[b & 3] += 1
    maxsuit = max(asuits)
    fh = fl = 0
    for c in hole:
        if asuits[c & 3] == maxsuit:
            r = (c >> 2) + 2
            if r > fh:
                fl, fh = fh, r
            else:
                fl = r
    cover, bcover, fourhigh = COVER[amask], COVER[mask], FOUR_HIGH[amask]
    sfcover = 0
    for s in {a & 3, b & 3}:
        sm = bmasks[s]
        for c in hole:
            if c & 3 == s:
                sm |= 1 << (c >> 2)
        sfcover = max(sfcover, COVER[sm])
    branks = [r for r in range(2, 15) if counts[r]]
    return [
        len(board), score >> 20,
        *[(score >> shift) & 15 for shift in (16, 12, 8, 4, 0)],
        bscore >> 20,
        *[(bscore >> shift) & 15 for shift in (16, 12, 8, 4, 0)],
        hi, lo, int(a & 3 == b & 3),
        sum(counts[r] for r in range(hi + 1, 15)),
        sum(counts[r] for r in range(lo + 1, 15)),
        counts[hi], counts[lo], max(counts), sum(n >= 2 for n in counts),
        max(bsuits), maxsuit, fh, fl, amask.bit_count(), cover, fourhigh,
        bcover, branks[-1], branks[0],
        max(hi if counts[hi] else 0, lo if counts[lo] else 0), int(hi == lo),
        len(branks), bsuits[a & 3], bsuits[b & 3],
        OUT_COUNT[amask], OUT_COUNT[mask], sfcover,
        max(bsuits) - min(bsuits),
        int((mask & (1 << (hi - 2))) != 0 and hi == lo),
        int(score == bscore),
    ]
