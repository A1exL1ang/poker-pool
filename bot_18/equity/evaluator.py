"""Independent standard Hold'em evaluator; larger integer means stronger hand.

Cards are ``4 * (rank - 2) + suit``. Five to seven cards are accepted. The
internal ``evaluate_partial`` also ranks a three/four-card board for features;
its result is not a completed poker hand. No third-party evaluator is used.
"""
from itertools import combinations


def _straight(mask):
    for high in range(12, 3, -1):
        if mask & (31 << (high - 4)) == 31 << (high - 4):
            return high + 2
    return 5 if mask & 0x100F == 0x100F else 0


STRAIGHT = tuple(_straight(mask) for mask in range(8192))
TOP5 = tuple(sum((r + 2) << (16 - 4 * i)
                 for i, r in enumerate([r for r in range(12, -1, -1)
                                        if mask & (1 << r)][:5]))
             for mask in range(8192))


def evaluate_partial(cards):
    counts = [0] * 13
    suits = [0] * 4
    mask = 0
    for c in cards:
        r = c >> 2
        bit = 1 << r
        counts[r] += 1
        suits[c & 3] |= bit
        mask |= bit
    flush = 0
    for sm in suits:
        if sm.bit_count() >= 5:
            high = STRAIGHT[sm]
            if high:
                return (8 << 20) | (high << 16)
            flush = max(flush, TOP5[sm])
    pairs = []
    trips = []
    quad = -1
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            quad = r
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)
    if quad >= 0:
        return (7 << 20) | ((quad + 2) << 16) | ((mask ^ (1 << quad)).bit_length() + 1) << 12
    if trips and (pairs or len(trips) > 1):
        pair = max(pairs[0] if pairs else -1, trips[1] if len(trips) > 1 else -1)
        return (6 << 20) | ((trips[0] + 2) << 16) | ((pair + 2) << 12)
    if flush:
        return (5 << 20) | flush
    high = STRAIGHT[mask]
    if high:
        return (4 << 20) | (high << 16)
    if trips:
        return (3 << 20) | ((trips[0] + 2) << 16) | ((TOP5[mask ^ (1 << trips[0])] >> 4) & 0xFF00)
    if len(pairs) >= 2:
        rest = mask ^ (1 << pairs[0]) ^ (1 << pairs[1])
        return (2 << 20) | ((pairs[0] + 2) << 16) | ((pairs[1] + 2) << 12) | ((rest.bit_length() + 1) << 8 if rest else 0)
    if pairs:
        return (1 << 20) | ((pairs[0] + 2) << 16) | ((TOP5[mask ^ (1 << pairs[0])] >> 4) & 0xFFF0)
    return TOP5[mask]


def evaluate(cards):
    """Evaluate five to seven distinct cards (input validation intentionally light)."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("evaluate requires five to seven cards")
    return evaluate_partial(cards)


def _reference_five(cards):
    """Slow independent count/sort implementation for correctness tests."""
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    groups = sorted(((ranks.count(r), r) for r in set(ranks)), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    unique = sorted(set(ranks))
    straight = 0
    if len(unique) == 5:
        if unique[-1] - unique[0] == 4:
            straight = unique[-1]
        elif unique == [2, 3, 4, 5, 14]:
            straight = 5
    if straight and flush:
        cat, keys = 8, [straight]
    elif groups[0][0] == 4:
        cat, keys = 7, [groups[0][1], groups[1][1]]
    elif [g[0] for g in groups] == [3, 2]:
        cat, keys = 6, [g[1] for g in groups]
    elif flush:
        cat, keys = 5, ranks
    elif straight:
        cat, keys = 4, [straight]
    elif groups[0][0] == 3:
        cat, keys = 3, [g[1] for g in groups]
    elif groups[0][0] == 2 and groups[1][0] == 2:
        cat, keys = 2, [g[1] for g in groups]
    elif groups[0][0] == 2:
        cat, keys = 1, [g[1] for g in groups]
    else:
        cat, keys = 0, ranks
    return (cat << 20) | sum(r << (16 - 4 * i) for i, r in enumerate(keys))


def evaluate_reference(cards):
    return max(_reference_five(c) for c in combinations(cards, 5))


def category(value):
    return value >> 20


def decode(value):
    return (value >> 20, *((value >> s) & 15 for s in (16, 12, 8, 4, 0)))
