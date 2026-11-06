"""Small, exact hero-only ICM calculation. All payouts are caller supplied."""


def hero_icm(stacks, hero, payouts):
    """ICM expected finishing points; zero stacks are already eliminated.

    A sole newly eliminated hero receives the lowest remaining finishing
    payout. Simultaneous eliminations must be resolved by the game adapter.
    Runtime is O(n*2**n), bounded here to eight original seats.
    """
    h = stacks[hero]
    others = [float(s) for i, s in enumerate(stacks) if i != hero and s > 0]
    n = len(others) + (h > 0)
    pay = list(payouts) + [0.0] * 8
    if h <= 0:
        return float(pay[len(others)])
    if not others:
        return float(pay[0])
    total = h + sum(others)
    size = 1 << len(others)
    sums = [0.0] * size
    probs = [0.0] * size
    probs[0] = 1.0
    value = 0.0
    for mask in range(size):
        if mask:
            low = mask & -mask
            sums[mask] = sums[mask ^ low] + others[low.bit_length() - 1]
        remainder = total - sums[mask]
        p = probs[mask]
        rank = mask.bit_count()
        value += p * h / remainder * pay[rank]
        if rank + 1 >= n or all(x == pay[rank] for x in pay[rank:n]):
            # The general recursion below is cheap; no approximation needed.
            if rank + 1 < n:
                value += p * (remainder - h) / remainder * pay[rank]
            continue
        remaining = (size - 1) ^ mask
        while remaining:
            low = remaining & -remaining
            j = low.bit_length() - 1
            probs[mask | low] += p * others[j] / remainder
            remaining ^= low
    return value


def call_threshold(obs, opponent, chip_threshold):
    """Exact ICM call threshold for an isolated heads-up all-in pot.

    Returns chip odds for ineligible states (side pots, incomplete payouts,
    non-all-in calls). No payout schedule is invented.
    """
    payouts = obs.get('payouts')
    if not payouts:
        return chip_threshold
    hero = obs['hero']
    stacks = obs['stacks']
    call = min(obs['to_call'], stacks[hero])
    if call <= 0 or (call < stacks[hero] and stacks[opponent] > 0):
        return chip_threshold
    contrib = obs['contributions']
    cap = contrib[hero] + call
    if any(c > cap for c in contrib):
        return chip_threshold
    # Remaining stacks plus the sole pot are the three exact terminal states.
    pot = sum(contrib)
    fold = list(stacks)
    fold[opponent] += pot
    lose = list(stacks)
    lose[hero] -= call
    lose[opponent] += pot + call
    win = list(stacks)
    win[hero] += pot
    uf = hero_icm(fold, hero, payouts)
    ul = hero_icm(lose, hero, payouts)
    uw = hero_icm(win, hero, payouts)
    if uw <= ul + 1e-12:
        return chip_threshold
    return max(chip_threshold, min(1.0, (uf - ul) / (uw - ul)))
