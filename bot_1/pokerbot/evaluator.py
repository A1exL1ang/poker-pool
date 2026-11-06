"""Exact 5--7 card evaluation using four suit bitboards, without dependencies.

Card integers are suit*13 + rank, rank 0=deuce, 12=ace; suits cdhs.
Larger scores are stronger. All kickers and the wheel are included.
"""
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
BITS = tuple(1 << c for c in range(52))
MASK = 8191
STRAIGHT = [0] * 8192
TOP1 = [0] * 8192
TOP2 = [0] * 8192
TOP3 = [0] * 8192
TOP5 = [0] * 8192
for _m in range(1, 8192):
    _rs = [r for r in range(12, -1, -1) if _m & (1 << r)]
    TOP1[_m] = _rs[0]
    for _k, _tab in ((2, TOP2), (3, TOP3), (5, TOP5)):
        _v = 0
        for _r in _rs[:_k]:
            _v = (_v << 4) | _r
        _tab[_m] = _v
    for _r in range(12, 3, -1):
        if _m & (31 << (_r - 4)) == 31 << (_r - 4):
            STRAIGHT[_m] = _r + 1
            break
    if not STRAIGHT[_m] and _m & 4111 == 4111:
        STRAIGHT[_m] = 4  # five high; values offset by one, zero means absent


def parse_card(card):
    if isinstance(card, int) and not isinstance(card, bool) and 0 <= card < 52:
        return card
    if isinstance(card, str):
        card = card.strip()
        if len(card) == 3 and card[:2] == "10":
            card = "T" + card[2]
        if len(card) == 2:
            return SUITS.index(card[1].lower()) * 13 + RANKS.index(card[0].upper())
    raise ValueError("card must be 0..51 or a string such as As or Td")


def card_name(card):
    return RANKS[card % 13] + SUITS[card // 13]


def packed(cards):
    bits = 0
    for c in cards:
        bits |= BITS[c]
    return bits


def evaluate_bits(bits):
    a = bits & 8191
    b = (bits >> 13) & 8191
    c = (bits >> 26) & 8191
    d = bits >> 39
    ranks = a | b | c | d
    flush = a if a.bit_count() >= 5 else b if b.bit_count() >= 5 else c if c.bit_count() >= 5 else d if d.bit_count() >= 5 else 0
    if flush:
        sf = STRAIGHT[flush]
        if sf:
            return 0x8000000 | sf
    four = a & b & c & d
    if four:
        q = TOP1[four]
        return 0x7000000 | (q << 4) | TOP1[ranks ^ (1 << q)]
    ab = a & b
    cd = c & d
    trips = (ab & (c | d)) | (cd & (a | b))
    pairs = ab | cd | ((a | b) & (c | d))
    if trips:
        t = TOP1[trips]
        rest = pairs ^ (1 << t)
        if rest:
            return 0x6000000 | (t << 4) | TOP1[rest]
    if flush:
        return 0x5000000 | TOP5[flush]
    straight = STRAIGHT[ranks]
    if straight:
        return 0x4000000 | straight
    if trips:
        return 0x3000000 | (t << 8) | TOP2[ranks ^ (1 << t)]
    if pairs:
        p = TOP1[pairs]
        other = pairs ^ (1 << p)
        if other:
            q = TOP1[other]
            return 0x2000000 | (p << 8) | (q << 4) | TOP1[ranks & ~(1 << p) & ~(1 << q)]
        return 0x1000000 | (p << 12) | TOP3[ranks ^ (1 << p)]
    return TOP5[ranks]


def evaluate(cards):
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError("evaluate requires 5..7 distinct integer cards")
    return evaluate_bits(packed(cards))


def reference_five(cards):
    """Deliberately independent, slow tuple evaluator for validation only."""
    ranks = sorted((c % 13 for c in cards), reverse=True)
    counts = {r: ranks.count(r) for r in set(ranks)}
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    unique = sorted(counts, reverse=True)
    straight = -1
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight = unique[0]
        elif unique == [12, 3, 2, 1, 0]:
            straight = 3
    flush = len({c // 13 for c in cards}) == 1
    if flush and straight >= 0:
        cat, tie = 8, [straight + 1]
    elif groups[0][0] == 4:
        cat, tie = 7, [groups[0][1], groups[1][1]]
    elif [n for n, _ in groups] == [3, 2]:
        cat, tie = 6, [groups[0][1], groups[1][1]]
    elif flush:
        cat, tie = 5, ranks
    elif straight >= 0:
        cat, tie = 4, [straight + 1]
    elif groups[0][0] == 3:
        cat, tie = 3, [groups[0][1]] + sorted((r for r in ranks if r != groups[0][1]), reverse=True)
    elif groups[0][0] == 2 and groups[1][0] == 2:
        ps = sorted((r for n, r in groups if n == 2), reverse=True)
        cat, tie = 2, ps + [r for n, r in groups if n == 1]
    elif groups[0][0] == 2:
        cat, tie = 1, [groups[0][1]] + sorted((r for r in ranks if r != groups[0][1]), reverse=True)
    else:
        cat, tie = 0, ranks
    value = cat << 24
    packed_tie = 0
    for r in tie:
        packed_tie = (packed_tie << 4) | r
    return value | packed_tie


def reference(cards):
    return max(reference_five(c) for c in combinations(cards, 5))
