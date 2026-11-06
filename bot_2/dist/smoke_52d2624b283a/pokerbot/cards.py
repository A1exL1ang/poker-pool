"""Compact, independently implemented 5-7 card evaluator.

Cards: rank * 4 + suit; ranks 2..A, suits c,d,h,s. Higher scores win.
All lookup tables are small, generated at import, and contain no game data.
"""

RANKS = "23456789TJQKA"
SUITS = "cdhs"
BITS = tuple(1 << (c >> 2) for c in range(52))
HIGH = [0] * 8192
TOP5 = [0] * 8192
TOP3 = [0] * 8192
TOP2 = [0] * 8192
STRAIGHT = [0] * 8192
COUNT = [0] * 8192
for _m in range(1, 8192):
    _rs = [r for r in range(12, -1, -1) if _m & (1 << r)]
    HIGH[_m] = _rs[0] + 2
    COUNT[_m] = len(_rs)
    _v = 0
    for _i, _r in enumerate(_rs[:5]):
        _v |= (_r + 2) << (16 - 4 * _i)
    TOP5[_m] = _v
    TOP3[_m] = _v & 0xFFF00
    TOP2[_m] = _v & 0xFF000
    for _r in range(12, 3, -1):
        if (_m >> (_r - 4)) & 31 == 31:
            STRAIGHT[_m] = _r + 2
            break
    else:
        if _m & 0x100F == 0x100F:
            STRAIGHT[_m] = 5
del _m, _rs, _v, _i, _r


def card(value):
    if isinstance(value, int):
        if 0 <= value < 52:
            return value
        raise ValueError("card integer must be in 0..51")
    s = str(value).strip()
    if len(s) == 3 and s[:2] == "10":
        s = "T" + s[-1]
    if len(s) != 2:
        raise ValueError("cards must look like As, Td, or 10h")
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


def notation(c):
    return RANKS[c >> 2] + SUITS[c & 3]


def evaluate(cards, bits=BITS, hi=HIGH, top5=TOP5, top3=TOP3,
             top2=TOP2, straight=STRAIGHT, count=COUNT):
    """Evaluate any five through seven distinct cards; inputs trusted internally."""
    one = two = three = four = 0
    s0 = s1 = s2 = s3 = 0
    for c in cards:
        b = bits[c]
        four |= three & b
        three |= two & b
        two |= one & b
        one |= b
        s = c & 3
        if s == 0:
            s0 |= b
        elif s == 1:
            s1 |= b
        elif s == 2:
            s2 |= b
        else:
            s3 |= b
    flush = s0 if count[s0] >= 5 else s1 if count[s1] >= 5 else s2 if count[s2] >= 5 else s3 if count[s3] >= 5 else 0
    if flush and straight[flush]:
        return 0x800000 | straight[flush] << 16
    if four:
        return 0x700000 | hi[four] << 16 | hi[one ^ four] << 12
    if three:
        t = hi[three]
        p = two ^ (1 << (t - 2))
        if p:
            return 0x600000 | t << 16 | hi[p] << 12
    if flush:
        return 0x500000 | top5[flush]
    if straight[one]:
        return 0x400000 | straight[one] << 16
    if three:
        return 0x300000 | hi[three] << 16 | top2[one ^ three] >> 4
    if two:
        if two & (two - 1):
            a = hi[two]
            rest = two ^ (1 << (a - 2))
            b = hi[rest]
            return 0x200000 | a << 16 | b << 12 | hi[one & ~(1 << (a - 2)) & ~(1 << (b - 2))] << 8
        return 0x100000 | hi[two] << 16 | top3[one ^ two] >> 4
    return top5[one]


def class_index(a, b):
    """169-class matrix: diagonal pairs, upper triangle suited."""
    r, s = a >> 2, b >> 2
    if r == s:
        return 13 * r + r
    lo, hi = (r, s) if r < s else (s, r)
    return lo * 13 + hi if (a & 3) == (b & 3) else hi * 13 + lo


def category(score):
    return score >> 20


def board_state(cards, bits=BITS):
    one = two = three = four = 0
    suits = [0, 0, 0, 0]
    for c in cards:
        b = bits[c]
        four |= three & b
        three |= two & b
        two |= one & b
        one |= b
        suits[c & 3] |= b
    return one, two, three, four, *suits


def evaluate_two(state, a, b, bits=BITS, hi=HIGH, top5=TOP5, top3=TOP3,
                 top2=TOP2, straight=STRAIGHT, count=COUNT):
    """Evaluate two private cards on a preprocessed 3-5 card board."""
    one, two, three, four, s0, s1, s2, s3 = state
    for c in (a, b):
        bit = bits[c]
        four |= three & bit
        three |= two & bit
        two |= one & bit
        one |= bit
        s = c & 3
        if s == 0:
            s0 |= bit
        elif s == 1:
            s1 |= bit
        elif s == 2:
            s2 |= bit
        else:
            s3 |= bit
    flush = s0 if count[s0] >= 5 else s1 if count[s1] >= 5 else s2 if count[s2] >= 5 else s3 if count[s3] >= 5 else 0
    if flush and straight[flush]:
        return 0x800000 | straight[flush] << 16
    if four:
        return 0x700000 | hi[four] << 16 | hi[one ^ four] << 12
    if three:
        t = hi[three]
        p = two ^ (1 << (t - 2))
        if p:
            return 0x600000 | t << 16 | hi[p] << 12
    if flush:
        return 0x500000 | top5[flush]
    if straight[one]:
        return 0x400000 | straight[one] << 16
    if three:
        return 0x300000 | hi[three] << 16 | top2[one ^ three] >> 4
    if two:
        if two & (two - 1):
            a = hi[two]
            rest = two ^ (1 << (a - 2))
            b = hi[rest]
            return 0x200000 | a << 16 | b << 12 | hi[one & ~(1 << (a - 2)) & ~(1 << (b - 2))] << 8
        return 0x100000 | hi[two] << 16 | top3[one ^ two] >> 4
    return top5[one]
