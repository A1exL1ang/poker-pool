"""Independent Chip Model (Malmuth-Harville) for up to 8 players.

`icm(stacks, payouts)` returns each player's expected payout. Players with zero
chips get zero. Payouts are ordered from first place down; missing places pay 0.

Exact evaluation uses a subset DP over the still-unranked players: the next
place index is determined by the subset size, so one pass over subsets in
decreasing size computes every player's expectation (2^n * n work). Subset
structure is precomputed per n so the hot loop is tight pure Python.
"""

from functools import lru_cache

_STRUCT = {}


def _struct(m: int):
    """For each popcount layer, list of (mask, [(bit index, mask without bit)...])."""
    s = _STRUCT.get(m)
    if s is None:
        layers = [[] for _ in range(m + 1)]
        for mask in range(1, 1 << m):
            bits = [(i, mask ^ (1 << i)) for i in range(m) if mask >> i & 1]
            layers[len(bits)].append((mask, bits))
        _STRUCT[m] = s = layers
    return s


def icm(stacks, payouts) -> list[float]:
    stacks = [float(s) for s in stacks]
    n = len(stacks)
    payouts = (list(payouts) + [0.0] * n)[:n]
    alive = [i for i in range(n) if stacks[i] > 0]
    m = len(alive)
    if m == 0:
        return [0.0] * n
    out = [0.0] * n
    if m == 1:
        out[alive[0]] = payouts[0]
        return out
    pays = payouts[:m]
    if all(p == 0 for p in pays):
        return out
    k = max(i + 1 for i, p in enumerate(pays) if p != 0)
    res = _icm_core(tuple(stacks[i] for i in alive), tuple(pays[:k]))
    for idx, i in enumerate(alive):
        out[i] = res[idx]
    return out


def _icm_core(s: tuple, pays: tuple) -> list[float]:
    m = len(s)
    k = len(pays)
    total = sum(s)
    if k == 1:
        return [pays[0] * x / total for x in s]
    layers = _struct(m)
    full = (1 << m) - 1
    prob = [0.0] * (1 << m)
    prob[full] = 1.0
    ev = [0.0] * m
    # subset sums
    sums = [0.0] * (1 << m)
    for mask in range(1, 1 << m):
        low = mask & -mask
        sums[mask] = sums[mask ^ low] + s[low.bit_length() - 1]
    for pop in range(m, m - k, -1):
        place = m - pop
        pay = pays[place]
        more = place + 1 < k
        for mask, bits in layers[pop]:
            p = prob[mask]
            if p == 0.0:
                continue
            q = p / sums[mask]
            if more:
                for i, nm in bits:
                    pi = q * s[i]
                    ev[i] += pi * pay
                    prob[nm] += pi
            else:
                for i, nm in bits:
                    ev[i] += q * s[i] * pay
    return ev


@lru_cache(maxsize=200000)
def icm_cached(stacks: tuple, payouts: tuple) -> tuple:
    return tuple(icm(stacks, payouts))


def icm_reference(stacks, payouts) -> list[float]:
    """Permutation-enumeration oracle for tests (small n only)."""
    from itertools import permutations
    n = len(stacks)
    payouts = list(payouts) + [0.0] * (n - len(payouts))
    alive = [i for i in range(n) if stacks[i] > 0]
    out = [0.0] * n
    k = min(len(alive), max([i + 1 for i, p in enumerate(payouts) if p] or [0]))
    for order in permutations(alive, k):
        p = 1.0
        remaining = sum(stacks[i] for i in alive)
        for i in order:
            p *= stacks[i] / remaining
            remaining -= stacks[i]
        for place, i in enumerate(order):
            out[i] += p * payouts[place]
    return out
