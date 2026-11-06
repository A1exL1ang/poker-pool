"""Independent rank-histogram Hold'em evaluator. Larger integer ranks win."""

_STRAIGHT = [0] * 8192
for _m in range(8192):
    for _r in range(12, 3, -1):
        if _m & (31 << (_r - 4)) == 31 << (_r - 4):
            _STRAIGHT[_m] = _r + 2
            break
    if not _STRAIGHT[_m] and _m & 4111 == 4111:
        _STRAIGHT[_m] = 5


def evaluate(cards):
    """Rank 5–7 valid, distinct cards. Inputs are validated by the adapter."""
    counts = [0] * 13
    suits = [0] * 4
    mask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suits[c & 3] |= 1 << r
        mask |= 1 << r
    flush = 0
    for s in suits:
        if s.bit_count() >= 5:
            if _STRAIGHT[s]:
                return 8 * 759375 + _STRAIGHT[s] * 50625
            flush = s
            break
    four = trip = pair = -1
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            four = r
        elif n == 3:
            if trip == -1:
                trip = r
            elif pair == -1:
                pair = r
        elif n == 2 and pair == -1:
            pair = r
    if four >= 0:
        remaining = mask & ~(1 << four)
        kicker = remaining.bit_length() + 1 if remaining else 0
        return 7 * 759375 + (four + 2) * 50625 + kicker * 3375
    if trip >= 0 and pair >= 0:
        return 6 * 759375 + (trip + 2) * 50625 + (pair + 2) * 3375
    if flush:
        value = 5 * 759375
        factor = 50625
        for r in range(12, -1, -1):
            if flush & (1 << r):
                value += (r + 2) * factor
                factor //= 15
                if not factor:
                    break
        return value
    if _STRAIGHT[mask]:
        return 4 * 759375 + _STRAIGHT[mask] * 50625
    groups = sorted(((n, r + 2) for r, n in enumerate(counts) if n), reverse=True)
    cat = 3 if groups[0][0] == 3 else (2 if len(groups) > 1 and groups[0][0] == groups[1][0] == 2 else (1 if groups[0][0] == 2 else 0))
    if cat == 2:
        a, b = groups[0][1], groups[1][1]
        kicker = max((r + 2 for r, n in enumerate(counts) if n and r + 2 not in (a, b)), default=0)
        return 2 * 759375 + a * 50625 + b * 3375 + kicker * 225
    value = cat * 759375
    factor = 50625
    for _, r in groups[:5 if cat == 0 else 4 if cat == 1 else 3]:
        value += r * factor
        factor //= 15
    return value


def category(rank):
    return rank // 759375
