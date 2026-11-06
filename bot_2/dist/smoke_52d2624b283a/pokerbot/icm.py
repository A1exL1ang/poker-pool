"""Exact independent chip model. No payout structure is assumed."""

from functools import lru_cache
from math import comb


@lru_cache(maxsize=64)
def _weights(payouts):
    m = len(payouts) - 1
    return tuple(sum((-1) ** j * comb(k, j) * payouts[m - k + j]
                     for j in range(k + 1)) for k in range(m + 1))


def focal_equity(stacks, hero, payouts):
    """Exact focal ICM via exponential-race inclusion/exclusion, O(2**(n-1))."""
    live = [i for i, s in enumerate(stacks) if s > 0]
    if hero not in live:
        # A hypothetical bust receives the next finishing-place payout.
        return payouts[min(len(live), len(payouts) - 1)]
    h = stacks[hero]
    others = [stacks[i] for i in live if i != hero]
    w = _weights(tuple(payouts[:len(live)]))
    sums = [0]
    sizes = [0]
    value = w[0]
    for s in others:
        length = len(sums)
        for j in range(length):
            t = sums[j] + s
            k = sizes[j] + 1
            sums.append(t)
            sizes.append(k)
            value += h / (h + t) * w[k]
    return value


def equities(stacks, payouts):
    """Expected points per seat under ICM; at most eight seats recommended."""
    n = len(stacks)
    if not n or any(s < 0 for s in stacks) or sum(stacks) <= 0:
        raise ValueError("positive total stack required")
    if len(payouts) < n:
        raise ValueError("one payout per stack required")
    result = [0.0] * n
    # DP over subsets: probability that this exact set filled the first k places.
    prob = [0.0] * (1 << n)
    prob[0] = 1.0
    sums = [0] * (1 << n)
    total = sum(stacks)
    for mask in range(1, 1 << n):
        bit = mask & -mask
        sums[mask] = sums[mask ^ bit] + stacks[bit.bit_length() - 1]
    for mask in range((1 << n) - 1):
        p = prob[mask]
        k = mask.bit_count()
        rem = total - sums[mask]
        remaining_count = n - k
        for i, stack in enumerate(stacks):
            bit = 1 << i
            if not mask & bit:
                q = p * (stack / rem if rem else 1 / remaining_count)
                result[i] += q * payouts[k]
                prob[mask | bit] += q
    return result


def bubble_factor(stacks, hero, villain, payouts, wager):
    """Local ICM loss/gain ratio for an equal-chip all-in confrontation."""
    wager = min(wager, stacks[hero], stacks[villain])
    if wager <= 0:
        return 1.0
    # Group subsets by whether they include villain. Since the two players
    # exchange chips, all with-villain denominators are identical in all three
    # scenarios. This computes the exact ratio in 2**(n-2) terms, not three full
    # ICM runs. The limit 0/0=1 for the empty subset gives the bust payout.
    live = [i for i, s in enumerate(stacks) if s > 0]
    weights = _weights(tuple(payouts[:len(live)]))
    h, v = stacks[hero], stacks[villain]
    up, down = h + wager, h - wager
    sums, sizes = [0], [0]
    for i in live:
        if i != hero and i != villain:
            length = len(sums)
            for j in range(length):
                sums.append(sums[j] + stacks[i])
                sizes.append(sizes[j] + 1)
    base = won = lost = 0.0
    for t, k in zip(sums, sizes):
        w = weights[k]
        shared = weights[k + 1] / (h + v + t)
        base += h / (h + t) * w + h * shared
        won += up / (up + t) * w + up * shared
        lost += (down / (down + t) if down + t else 1.0) * w + down * shared
    gain = won - base
    loss = base - lost
    return max(1.0, min(100.0, loss / max(gain, 1e-12)))
