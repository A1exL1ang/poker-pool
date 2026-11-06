"""Small, exact, allocation-light Texas Hold'em hand evaluator.

Cards use ``4 * (rank - 2) + suit`` (rank 2..14, suit 0..3).
``evaluate`` accepts five through seven unique cards. Higher ranks win. The
category is the top nibble (``value >> 20``), from high card (0) to straight
flush (8). The next five nibbles hold descending tie breakers. Callers validate
card identities at the observation boundary; the hot evaluator trusts them.
"""

from __future__ import annotations


def _make_tables():
    straights = [0] * 8192
    kickers = [0] * 8192
    for mask in range(1, 8192):
        bits = mask
        packed = 0
        shift = 16
        while bits and shift >= 0:
            r = bits.bit_length() - 1
            packed |= (r + 2) << shift
            bits ^= 1 << r
            shift -= 4
        kickers[mask] = packed
        for high in range(12, 3, -1):
            run = 31 << (high - 4)
            if mask & run == run:
                straights[mask] = high + 2
                break
        else:
            if mask & 0x100F == 0x100F:
                straights[mask] = 5
    return tuple(straights), tuple(kickers)


STRAIGHT_HIGH, TOP_FIVE = _make_tables()
del _make_tables
RANK_BITS = tuple(1 << (card >> 2) for card in range(52))


def evaluate(cards) -> int:
    """Return the exact best-five rank of five to seven legal cards."""
    ranks = pairs = trips = quads = 0
    suits = [0, 0, 0, 0]
    for card in cards:
        bit = RANK_BITS[card]
        quads |= trips & bit
        trips |= pairs & bit
        pairs |= ranks & bit
        ranks |= bit
        suits[card & 3] |= bit
    flush = 0
    for suit in suits:
        if suit.bit_count() >= 5:
            straight = STRAIGHT_HIGH[suit]
            if straight:
                return 0x800000 | (straight << 16)
            flush = TOP_FIVE[suit]
            break
    if quads:
        q = quads.bit_length() - 1
        return 0x700000 | ((q + 2) << 16) | ((ranks ^ (1 << q)).bit_length() + 1) << 12
    if trips:
        t = trips.bit_length() - 1
        others = pairs ^ (1 << t)
        if others:
            return 0x600000 | ((t + 2) << 16) | ((others.bit_length() + 1) << 12)
    if flush:
        return 0x500000 | flush
    straight = STRAIGHT_HIGH[ranks]
    if straight:
        return 0x400000 | (straight << 16)
    if trips:
        return 0x300000 | ((t + 2) << 16) | ((TOP_FIVE[ranks ^ (1 << t)] >> 4) & 0xFF00)
    if pairs:
        p = pairs.bit_length() - 1
        other_pairs = pairs ^ (1 << p)
        if other_pairs:
            q = other_pairs.bit_length() - 1
            kicker = (ranks ^ (1 << p) ^ (1 << q)).bit_length() + 1
            return 0x200000 | ((p + 2) << 16) | ((q + 2) << 12) | (kicker << 8)
        return 0x100000 | ((p + 2) << 16) | ((TOP_FIVE[ranks ^ (1 << p)] >> 4) & 0xFFF0)
    return TOP_FIVE[ranks]


def card_id(text: str) -> int:
    """Parse conventional ``As``/``Td`` card text; suits are c,d,h,s."""
    text = text.strip()
    if len(text) == 3 and text[:2] == "10":
        text = "T" + text[2]
    if len(text) != 2:
        raise ValueError("Card must contain a rank and a suit")
    return "23456789TJQKA".index(text[0].upper()) * 4 + "cdhs".index(text[1].lower())


def card_text(card: int) -> str:
    if not isinstance(card, int) or not 0 <= card < 52:
        raise ValueError("Card ID must be in 0..51")
    return "23456789TJQKA"[card >> 2] + "cdhs"[card & 3]
