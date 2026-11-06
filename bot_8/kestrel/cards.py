"""Small, dependency-free 5--7-card evaluator. Higher scores win.

Cards are 4 * (rank - 2) + suit; suits are clubs, diamonds, hearts, spades.
The score is category << 20 followed by five hexadecimal kicker digits.
All tables are constructed once on import, outside the decision path.
"""

RANKS = "23456789TJQKA"
SUITS = "cdhs"
BITS = tuple(1 << (c >> 2) for c in range(52))
TOP = [0] * 8192
STRAIGHT = [0] * 8192
for _mask in range(1, 8192):
    _r = [r + 2 for r in range(12, -1, -1) if _mask & (1 << r)]
    _v = 0
    for _i, _rank in enumerate(_r[:5]):
        _v |= _rank << (16 - 4 * _i)
    TOP[_mask] = _v
    for _high in range(12, 3, -1):
        if _mask & (31 << (_high - 4)) == 31 << (_high - 4):
            STRAIGHT[_mask] = _high + 2
            break
    if not STRAIGHT[_mask] and _mask & 0x100F == 0x100F:
        STRAIGHT[_mask] = 5
TOP = tuple(TOP)
STRAIGHT = tuple(STRAIGHT)
EMPTY = (0, 0, 0, 0, 0, 0, 0, 0)


def card(value):
    """Parse canonical integers or case-insensitive strings such as 'As'."""
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 52:
        return value
    if isinstance(value, str):
        value = value.strip()
        if len(value) == 3 and value[:2] == "10":
            value = "T" + value[-1]
        if len(value) == 2:
            try:
                return 4 * RANKS.index(value[0].upper()) + SUITS.index(value[1].lower())
            except ValueError:
                pass
    raise ValueError(f"Invalid card: {value!r}")


def text(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def extend(state, cards):
    ranks, pairs, trips, quads, s0, s1, s2, s3 = state
    for c in cards:
        b = BITS[c]
        quads |= trips & b
        trips |= pairs & b
        pairs |= ranks & b
        ranks |= b
        suit = c & 3
        if suit == 0:
            s0 |= b
        elif suit == 1:
            s1 |= b
        elif suit == 2:
            s2 |= b
        else:
            s3 |= b
    return ranks, pairs, trips, quads, s0, s1, s2, s3


def score(state):
    ranks, pairs, trips, quads, s0, s1, s2, s3 = state
    flush = 0
    if s0.bit_count() >= 5:
        flush = s0
    elif s1.bit_count() >= 5:
        flush = s1
    elif s2.bit_count() >= 5:
        flush = s2
    elif s3.bit_count() >= 5:
        flush = s3
    if flush:
        straight = STRAIGHT[flush]
        if straight:
            return (8 << 20) | (straight << 16)
    if quads:
        q = quads.bit_length() - 1
        k = (ranks ^ (1 << q)).bit_length() + 1
        return (7 << 20) | ((q + 2) << 16) | (k << 12)
    if trips:
        t = trips.bit_length() - 1
        remaining_pairs = pairs ^ (1 << t)
        if remaining_pairs:
            p = remaining_pairs.bit_length() + 1
            return (6 << 20) | ((t + 2) << 16) | (p << 12)
    if flush:
        return (5 << 20) | TOP[flush]
    straight = STRAIGHT[ranks]
    if straight:
        return (4 << 20) | (straight << 16)
    if trips:
        return (3 << 20) | ((t + 2) << 16) | ((TOP[ranks ^ (1 << t)] >> 4) & 0xFF00)
    if pairs:
        p = pairs.bit_length() - 1
        rest = pairs ^ (1 << p)
        if rest:
            p2 = rest.bit_length() - 1
            k = (ranks & ~(1 << p) & ~(1 << p2)).bit_length() + 1
            return (2 << 20) | ((p + 2) << 16) | ((p2 + 2) << 12) | (k << 8)
        return (1 << 20) | ((p + 2) << 16) | ((TOP[ranks ^ (1 << p)] >> 4) & 0xFFF0)
    return TOP[ranks]


def evaluate(cards):
    """Evaluate 5--7 distinct integer cards; internal callers validate once."""
    return score(extend(EMPTY, cards))


def hand_class(a, b):
    """169-class matrix: pairs diagonal; suited above, offsuit below."""
    ra, rb = a >> 2, b >> 2
    if ra == rb:
        return 13 * ra + ra
    hi, lo = (ra, rb) if ra > rb else (rb, ra)
    return 13 * lo + hi if (a & 3) == (b & 3) else 13 * hi + lo


def class_name(index):
    a, b = divmod(index, 13)
    if a == b:
        return RANKS[a] * 2
    return RANKS[max(a, b)] + RANKS[min(a, b)] + ("s" if a < b else "o")


def draw_strength(hole, board):
    """Cheap draw descriptor: flush draw, straight draw, nut-flush blocker.

    Only draws that use at least one hole card are counted. This is a policy
    feature, not an outs-to-equity calculator (overlapping outs are possible).
    """
    if len(board) == 5:
        return 0.0, False
    mask = 0
    board_mask = 0
    suits = [0, 0, 0, 0]
    for c in board:
        mask |= BITS[c]
        board_mask |= BITS[c]
        suits[c & 3] += 1
    for c in hole:
        mask |= BITS[c]
        suits[c & 3] += 1
    flush = any(suits[c & 3] == 4 for c in hole)
    nut_draw = any((c >> 2) == 12 and suits[c & 3] >= 3 for c in hole)
    outs = set()
    for window in (0x100F, 31, 62, 124, 248, 496, 992, 1984, 3968, 7936):
        hit = mask & window
        if hit.bit_count() == 4 and (hit & ~board_mask):
            outs.add(window ^ hit)
    straight = min(len(outs), 2)
    strength = (0.34 if flush else 0.0) + 0.13 * straight
    if len(board) == 4:
        strength *= 0.56
    return min(strength, 0.56), nut_draw
