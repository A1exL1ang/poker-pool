"""Exact subset ICM, for user-supplied points only (no assumed contest payouts)."""


def hero_equity(stacks, payouts, hero):
    """Expected finishing points under sequential chip-proportional selection.

    Zero stacks share the trailing prizes equally. Existing eliminated players
    should be removed before calling if their finishing prizes are already paid.
    Complexity O(n 2**(n-1)); intended for n <= 8.
    """
    n = len(stacks)
    if not n or not 0 <= hero < n:
        raise ValueError("invalid hero/stacks")
    if len(payouts) < n or any(s < 0 for s in stacks):
        raise ValueError("invalid payouts/stacks")
    positive = [i for i, s in enumerate(stacks) if s > 0]
    if stacks[hero] == 0:
        trailing = payouts[len(positive):n]
        return sum(trailing) / len(trailing)
    own = float(stacks[hero])
    others = [float(stacks[i]) for i in positive if i != hero]
    total = own + sum(others)
    # Winner-take-all and equal-prize cases also avoid needless computation.
    m = len(positive)
    if m == 1:
        return float(payouts[0])
    if all(p == payouts[m - 1] for p in payouts[1:m]):
        return payouts[m - 1] + (payouts[0] - payouts[m - 1]) * own / total
    size = 1 << len(others)
    probability = [0.0] * size
    removed = [0.0] * size
    probability[0] = 1.0
    result = 0.0
    for mask in range(size):
        if mask:
            low = mask & -mask
            removed[mask] = removed[mask ^ low] + others[low.bit_length() - 1]
        remaining = total - removed[mask]
        p = probability[mask] / remaining
        result += p * own * payouts[mask.bit_count()]
        available = (size - 1) ^ mask
        while available:
            bit = available & -available
            probability[mask | bit] += p * others[bit.bit_length() - 1]
            available ^= bit
    return result


def equities(stacks, payouts):
    return [hero_equity(stacks, payouts, i) for i in range(len(stacks))]
