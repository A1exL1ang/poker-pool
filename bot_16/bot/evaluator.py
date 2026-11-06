"""Exact five-to-seven-card rank evaluation, implemented independently.

Higher scores win. Category is score >> 20; five rank nibbles break ties.
Ace is 14, except a wheel's straight high card is 5. No suit breaks a tie.
Lookup tables cover rank masks only and are generated locally at import.
"""


def _tables():
    top = [0] * 8192
    straight = [0] * 8192
    for mask in range(8192):
        m, value, shift = mask, 0, 16
        while m and shift >= 0:
            bit = m.bit_length() - 1
            value |= (bit + 2) << shift
            m ^= 1 << bit
            shift -= 4
        top[mask] = value
        for high in range(12, 3, -1):
            run = 31 << (high - 4)
            if mask & run == run:
                straight[mask] = high + 2
                break
        if not straight[mask] and mask & 0x100F == 0x100F:
            straight[mask] = 5
    return top, straight


TOP, STRAIGHT = _tables()
BITS = tuple(1 << (c // 4) for c in range(52))
SUITS = tuple(c % 4 for c in range(52))


def evaluate(cards):
    """Return exact best five-card score for 5..7 distinct integer cards.

    Partial 2..4-card hands return their current made-hand category for features.
    Input validity is enforced at the observation boundary, not in this hot path.
    """
    once = twice = thrice = four = 0
    suits = [0, 0, 0, 0]
    for card in cards:
        bit = BITS[card]
        four |= thrice & bit
        thrice |= twice & bit
        twice |= once & bit
        once |= bit
        suits[SUITS[card]] |= bit
    flush = 0
    for mask in suits:
        if mask.bit_count() >= 5:
            high = STRAIGHT[mask]
            if high:
                return (8 << 20) | (high << 16)
            flush = TOP[mask]
            break
    if four:
        high = four.bit_length() - 1
        kicker = (once ^ (1 << high)).bit_length() + 1
        return (7 << 20) | ((high + 2) << 16) | (kicker << 12)
    if thrice:
        high = thrice.bit_length() - 1
        rest = twice ^ (1 << high)
        if rest:
            return (6 << 20) | ((high + 2) << 16) | ((rest.bit_length() + 1) << 12)
    if flush:
        return (5 << 20) | flush
    high = STRAIGHT[once]
    if high:
        return (4 << 20) | (high << 16)
    if thrice:
        bit = thrice.bit_length() - 1
        return (3 << 20) | ((bit + 2) << 16) | ((TOP[once ^ (1 << bit)] >> 4) & 0xFF00)
    if twice:
        first = twice.bit_length() - 1
        rest = twice ^ (1 << first)
        if rest:
            second = rest.bit_length() - 1
            kicker = (once & ~((1 << first) | (1 << second))).bit_length() + 1
            return (2 << 20) | ((first + 2) << 16) | ((second + 2) << 12) | (kicker << 8)
        return (1 << 20) | ((first + 2) << 16) | ((TOP[once ^ (1 << first)] >> 4) & 0xFFF0)
    return TOP[once]


def card(text):
    """Human-readable test helper, e.g. 'As' -> 51."""
    return '23456789TJQKA'.index(text[0].upper()) * 4 + 'cdhs'.index(text[1].lower())
