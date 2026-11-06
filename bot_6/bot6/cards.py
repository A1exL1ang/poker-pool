"""Exact 5–7 card evaluator using four 13-bit suit masks.

Cards are rank-major integers: 0=2c, 1=2d, ..., 51=As.
Larger scores are stronger; equal scores split the pot. No third-party code.
"""
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CARD_MASK = tuple(1 << ((c % 4) * 13 + c // 4) for c in range(52))
HIGH = [0] * 8192
TOP = [0] * 8192
STRAIGHT = [0] * 8192
for _mask in range(1, 8192):
    _ranks = [r + 2 for r in range(12, -1, -1) if _mask & (1 << r)]
    HIGH[_mask] = _ranks[0]
    TOP[_mask] = sum(r << (20 - 4 * i) for i, r in enumerate(_ranks[:5]))
    for _hi in range(12, 3, -1):
        if (_mask >> (_hi - 4)) & 31 == 31:
            STRAIGHT[_mask] = _hi + 2
            break
    if not STRAIGHT[_mask] and _mask & 4111 == 4111:
        STRAIGHT[_mask] = 5
RANK_BIT = tuple(0 if r < 2 else 1 << (r - 2) for r in range(15))


def parse_card(card):
    if isinstance(card, int) and not isinstance(card, bool):
        if 0 <= card < 52:
            return card
        raise ValueError("Card integer must be in 0..51")
    if isinstance(card, str):
        card = card.strip().replace("10", "T")
        if len(card) == 2 and card[0].upper() in RANKS and card[1].lower() in SUITS:
            return RANKS.index(card[0].upper()) * 4 + SUITS.index(card[1].lower())
    raise ValueError("Expected card like As, Td, or rank-major integer 0..51")


def card_name(card):
    return RANKS[card // 4] + SUITS[card % 4]


def hand_class(a, b):
    ra, rb = a // 4, b // 4
    if ra < rb:
        ra, rb = rb, ra
    return RANKS[ra] + RANKS[rb] + ("" if ra == rb else "s" if a % 4 == b % 4 else "o")


def evaluate_mask(mask, high=HIGH, top=TOP, straight=STRAIGHT, bit=RANK_BIT):
    s0 = mask & 8191
    s1 = (mask >> 13) & 8191
    s2 = (mask >> 26) & 8191
    s3 = mask >> 39
    union = s0 | s1 | s2 | s3
    flush = s0 if s0.bit_count() >= 5 else s1 if s1.bit_count() >= 5 else s2 if s2.bit_count() >= 5 else s3 if s3.bit_count() >= 5 else 0
    if flush and straight[flush]:
        return (8 << 24) | (straight[flush] << 20)
    a, b = s0 & s1, s2 & s3
    c, d = s0 | s1, s2 | s3
    quads = a & b
    if quads:
        q = high[quads]
        return (7 << 24) | (q << 20) | (high[union ^ bit[q]] << 16)
    trips = (a & d) | (b & c)
    pairs = a | b | (c & d)
    if trips:
        t = high[trips]
        other = pairs ^ bit[t]
        if other:
            return (6 << 24) | (t << 20) | (high[other] << 16)
    if flush:
        return (5 << 24) | top[flush]
    if straight[union]:
        return (4 << 24) | (straight[union] << 20)
    if trips:
        return (3 << 24) | (t << 20) | ((top[union ^ bit[t]] >> 4) & 0xFF000)
    if pairs:
        p = high[pairs]
        other = pairs ^ bit[p]
        if other:
            p2 = high[other]
            return (2 << 24) | (p << 20) | (p2 << 16) | (high[union ^ bit[p] ^ bit[p2]] << 12)
        return (1 << 24) | (p << 20) | ((top[union ^ bit[p]] >> 4) & 0xFFF00)
    return top[union]


def evaluate(cards):
    cards = tuple(parse_card(c) for c in cards)
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError("Evaluator needs 5–7 distinct cards")
    return evaluate_mask(sum(CARD_MASK[c] for c in cards))


def reference_five(cards):
    """Slow, structurally independent reference used only by tests."""
    from collections import Counter
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    groups = sorted(((n, r) for r, n in Counter(ranks).items()), reverse=True)
    unique = sorted(set(ranks), reverse=True)
    st = unique[0] if len(unique) == 5 and unique[0] - unique[4] == 4 else 5 if unique == [14, 5, 4, 3, 2] else 0
    fl = len({c % 4 for c in cards}) == 1
    if st and fl:
        cat, kick = 8, [st]
    elif groups[0][0] == 4:
        cat, kick = 7, [groups[0][1], groups[1][1]]
    elif [g[0] for g in groups] == [3, 2]:
        cat, kick = 6, [groups[0][1], groups[1][1]]
    elif fl:
        cat, kick = 5, ranks
    elif st:
        cat, kick = 4, [st]
    elif groups[0][0] == 3:
        cat, kick = 3, [r for n, r in groups]
    elif [g[0] for g in groups[:2]] == [2, 2]:
        cat, kick = 2, [r for n, r in groups]
    elif groups[0][0] == 2:
        cat, kick = 1, [r for n, r in groups]
    else:
        cat, kick = 0, ranks
    return (cat << 24) | sum(r << (20 - 4 * i) for i, r in enumerate(kick))


def reference(cards):
    return max(reference_five(c) for c in combinations(cards, 5))
