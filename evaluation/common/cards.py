"""Card encoding and a fast 7-card hand evaluator.

Cards are ints 0..51: rank = card >> 2 (0 = deuce .. 12 = ace), suit = card & 3.
This equals rank = card // 4 + 2, suit = card % 4, so the encoding is shared
with the other local tooling in this repository.

Hand scores are ints; larger is better. The category occupies the high bits so
scores compare across categories.
"""

from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "cdhs"

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_SHIFT = 26
CATEGORY_NAMES = (
    "high card", "pair", "two pair", "three of a kind", "straight",
    "flush", "full house", "four of a kind", "straight flush",
)


def card(text: str) -> int:
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def cards(text: str) -> tuple[int, ...]:
    text = text.replace(" ", "")
    return tuple(card(text[i:i + 2]) for i in range(0, len(text), 2))


def card_str(c: int) -> str:
    return RANKS[c >> 2] + SUITS[c & 3]


def cards_str(cs) -> str:
    return " ".join(card_str(c) for c in cs)


def _build_tables():
    straight_hi = [-1] * 8192
    top5 = [0] * 8192
    top3 = [0] * 8192
    top2 = [0] * 8192
    top1 = [0] * 8192
    for m in range(8192):
        hi = -1
        for top in range(12, 3, -1):
            need = 0b11111 << (top - 4)
            if m & need == need:
                hi = top
                break
        if hi < 0 and (m & 0b1000000001111) == 0b1000000001111:
            hi = 3  # wheel
        straight_hi[m] = hi
        bits = [b for b in range(12, -1, -1) if m >> b & 1]
        top5[m] = sum(1 << b for b in bits[:5])
        top3[m] = sum(1 << b for b in bits[:3])
        top2[m] = sum(1 << b for b in bits[:2])
        top1[m] = sum(1 << b for b in bits[:1])
    return straight_hi, top5, top3, top2, top1


STRAIGHT_HI, TOP5, TOP3, TOP2, TOP1 = _build_tables()
_SF = STRAIGHT_FLUSH << CATEGORY_SHIFT
_QUADS = QUADS << CATEGORY_SHIFT
_FULL = FULL_HOUSE << CATEGORY_SHIFT
_FLUSH = FLUSH << CATEGORY_SHIFT
_STRAIGHT = STRAIGHT << CATEGORY_SHIFT
_TRIPS = TRIPS << CATEGORY_SHIFT
_TWO_PAIR = TWO_PAIR << CATEGORY_SHIFT
_PAIR = PAIR << CATEGORY_SHIFT


def evaluate(cs) -> int:
    """Score the best 5-card hand among 5..7 cards. Larger is better."""
    s0 = s1 = s2 = s3 = 0
    mask = 0
    cnt = [0] * 13
    for c in cs:
        r = c >> 2
        bit = 1 << r
        cnt[r] += 1
        mask |= bit
        s = c & 3
        if s == 0:
            s0 |= bit
        elif s == 1:
            s1 |= bit
        elif s == 2:
            s2 |= bit
        else:
            s3 |= bit
    for fm in (s0, s1, s2, s3):
        if fm.bit_count() >= 5:
            sh = STRAIGHT_HI[fm]
            if sh >= 0:
                return _SF | sh
            return _FLUSH | TOP5[fm]
    quad = trips = -1
    pair_hi = pair_lo = -1
    for r in range(12, -1, -1):
        n = cnt[r]
        if n == 4:
            quad = r
        elif n == 3:
            if trips < 0:
                trips = r
            elif pair_hi < 0:
                pair_hi = r
        elif n == 2:
            if pair_hi < 0:
                pair_hi = r
            elif pair_lo < 0:
                pair_lo = r
    if quad >= 0:
        return _QUADS | quad << 4 | TOP1[mask & ~(1 << quad)].bit_length() - 1
    if trips >= 0 and pair_hi >= 0:
        return _FULL | trips << 4 | pair_hi
    sh = STRAIGHT_HI[mask]
    if sh >= 0:
        return _STRAIGHT | sh
    if trips >= 0:
        return _TRIPS | trips << 13 | TOP2[mask & ~(1 << trips)]
    if pair_lo >= 0:
        return _TWO_PAIR | pair_hi << 17 | pair_lo << 13 | TOP1[mask & ~(1 << pair_hi) & ~(1 << pair_lo)]
    if pair_hi >= 0:
        return _PAIR | pair_hi << 13 | TOP3[mask & ~(1 << pair_hi)]
    return TOP5[mask]


def category(score: int) -> int:
    return score >> CATEGORY_SHIFT


def evaluate5_reference(cs):
    """Slow, obviously-correct 5-card scorer used only by tests."""
    ranks = sorted((c >> 2 for c in cs), reverse=True)
    suits = {c & 3 for c in cs}
    counts = {r: ranks.count(r) for r in set(ranks)}
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len(suits) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_hi = -1
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_hi = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_hi = 3
    if flush and straight_hi >= 0:
        return (STRAIGHT_FLUSH, straight_hi)
    if groups[0][1] == 4:
        return (QUADS, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (FULL_HOUSE, groups[0][0], groups[1][0])
    if flush:
        return (FLUSH, *ranks)
    if straight_hi >= 0:
        return (STRAIGHT, straight_hi)
    if groups[0][1] == 3:
        return (TRIPS, groups[0][0], *[g[0] for g in groups[1:]])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (TWO_PAIR, groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2:
        return (PAIR, groups[0][0], *[g[0] for g in groups[1:]])
    return (HIGH_CARD, *ranks)


def evaluate_reference(cs):
    return max(evaluate5_reference(five) for five in combinations(cs, 5))


# ---- preflop hand classes -------------------------------------------------
# 169 classes: index = hi*13+lo for suited (hi>lo), lo*13+hi for offsuit, r*13+r for pairs.

def hand_class(c1: int, c2: int) -> int:
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 == r2:
        return r1 * 13 + r1
    hi, lo = (r1, r2) if r1 > r2 else (r2, r1)
    if (c1 & 3) == (c2 & 3):
        return hi * 13 + lo
    return lo * 13 + hi


def class_name(idx: int) -> str:
    a, b = divmod(idx, 13)
    if a == b:
        return RANKS[a] * 2
    if a > b:
        return RANKS[a] + RANKS[b] + "s"
    return RANKS[b] + RANKS[a] + "o"


def class_index(name: str) -> int:
    hi, lo = RANKS.index(name[0]), RANKS.index(name[1])
    if hi < lo:
        hi, lo = lo, hi
    if hi == lo:
        return hi * 13 + hi
    if name[2] == "s":
        return hi * 13 + lo
    return lo * 13 + hi


def class_combos(idx: int) -> list[tuple[int, int]]:
    a, b = divmod(idx, 13)
    out = []
    if a == b:
        for s1 in range(4):
            for s2 in range(s1 + 1, 4):
                out.append((a * 4 + s1, a * 4 + s2))
    elif a > b:
        for s in range(4):
            out.append((a * 4 + s, b * 4 + s))
    else:
        for s1 in range(4):
            for s2 in range(4):
                if s1 != s2:
                    out.append((b * 4 + s1, a * 4 + s2))
    return out


CLASS_WEIGHT = [len(class_combos(i)) for i in range(169)]


# ---- faster evaluator: prime-product lookup for the non-flush case ----------
_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
PRIME = [_PRIMES[c >> 2] for c in range(52)]
SUITBIT = [1 << ((c & 3) * 13 + (c >> 2)) for c in range(52)]
_POP13 = [bin(i).count("1") for i in range(8192)]


def _build_rank_table():
    """Map prime product of 5..7 ranks (ignoring suits) -> non-flush score."""
    from itertools import combinations_with_replacement
    table = {}
    for k in (5, 6, 7):
        for combo in combinations_with_replacement(range(13), k):
            if any(combo.count(r) > 4 for r in set(combo)):
                continue
            # build concrete cards with distinct suits per rank, avoiding flushes
            cs = []
            used = {}
            for r in combo:
                s = used.get(r, 0)
                used[r] = s + 1
                cs.append(r * 4 + s)
            # ensure no flush: with <=4 per rank, suits cycle 0..3 so a flush needs
            # 5 ranks sharing suit 0 -> possible. Compute non-flush score directly.
            p = 1
            for r in combo:
                p *= _PRIMES[r]
            table[p] = _nonflush_score(combo)
    return table


def _nonflush_score(ranks):
    cnt = [0] * 13
    mask = 0
    for r in ranks:
        cnt[r] += 1
        mask |= 1 << r
    quad = trips = -1
    pair_hi = pair_lo = -1
    for r in range(12, -1, -1):
        n = cnt[r]
        if n == 4:
            quad = r
        elif n == 3:
            if trips < 0:
                trips = r
            elif pair_hi < 0:
                pair_hi = r
        elif n == 2:
            if pair_hi < 0:
                pair_hi = r
            elif pair_lo < 0:
                pair_lo = r
    if quad >= 0:
        return _QUADS | quad << 4 | TOP1[mask & ~(1 << quad)].bit_length() - 1
    if trips >= 0 and pair_hi >= 0:
        return _FULL | trips << 4 | pair_hi
    sh = STRAIGHT_HI[mask]
    if sh >= 0:
        return _STRAIGHT | sh
    if trips >= 0:
        return _TRIPS | trips << 13 | TOP2[mask & ~(1 << trips)]
    if pair_lo >= 0:
        return _TWO_PAIR | pair_hi << 17 | pair_lo << 13 | TOP1[mask & ~(1 << pair_hi) & ~(1 << pair_lo)]
    if pair_hi >= 0:
        return _PAIR | pair_hi << 13 | TOP3[mask & ~(1 << pair_hi)]
    return TOP5[mask]


RANK_TABLE = _build_rank_table()
_FLUSH_SCORE = [0] * 8192
for _m in range(8192):
    if _POP13[_m] >= 5:
        _sh = STRAIGHT_HI[_m]
        _FLUSH_SCORE[_m] = (_SF | _sh) if _sh >= 0 else (_FLUSH | TOP5[_m])


def evaluate_fast(cs) -> int:
    """Same ordering as evaluate(); ~3x faster. 5..7 cards."""
    p = 1
    m = 0
    for c in cs:
        p *= PRIME[c]
        m |= SUITBIT[c]
    if len(cs) >= 5:
        f = m & 8191
        if _POP13[f] >= 5:
            return _FLUSH_SCORE[f]
        f = m >> 13 & 8191
        if _POP13[f] >= 5:
            return _FLUSH_SCORE[f]
        f = m >> 26 & 8191
        if _POP13[f] >= 5:
            return _FLUSH_SCORE[f]
        f = m >> 39
        if _POP13[f] >= 5:
            return _FLUSH_SCORE[f]
    return RANK_TABLE[p]
