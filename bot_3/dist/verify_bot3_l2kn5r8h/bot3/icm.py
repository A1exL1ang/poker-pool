"""Exact independent chip model, solely for supplied payout vectors.

No default tournament payout structure is presumed. ICM is an approximation to
future tournament equity even though the arithmetic here is exact.
"""


def equity(stacks, payouts, hero):
    n = len(stacks)
    if not 0 <= hero < n or n > 8 or any(s < 0 for s in stacks):
        raise ValueError("ICM requires 1..8 nonnegative stacks and a valid hero")
    # An already eliminated hero receives the next finishing-place prize.
    if stacks[hero] == 0:
        alive = sum(s > 0 for s in stacks)
        return float(payouts[alive]) if alive < len(payouts) else 0.0
    live = [(i, float(s)) for i, s in enumerate(stacks) if s > 0]
    total = sum(s for _, s in live)
    opp = [s for i, s in live if i != hero]
    h = float(stacks[hero])
    # dp[mask]: probability that precisely this set of opponents finishes first,
    # in any order, before hero. Hero is selected next with h / remaining.
    size = 1 << len(opp)
    dp = [0.0] * size
    sums = [0.0] * size
    dp[0] = 1.0
    answer = 0.0
    for m in range(size):
        if m:
            bit = m & -m
            sums[m] = sums[m ^ bit] + opp[bit.bit_length() - 1]
        rem = total - sums[m]
        place = m.bit_count()
        if place < len(payouts):
            answer += dp[m] * h / rem * payouts[place]
        for j, s in enumerate(opp):
            bit = 1 << j
            if not m & bit:
                dp[m | bit] += dp[m] * s / rem
    return answer


def call_threshold(stacks, payouts, hero, villain, call, pot):
    """Binary, heads-up call approximation with the existing pot awarded whole.

    stacks are chips behind now; pot already includes the villain's wager.
    This deliberately does not pretend to solve a multiway side-pot ICM game.
    """
    if call <= 0:
        return 0.0
    folded, won, lost = list(stacks), list(stacks), list(stacks)
    folded[villain] += pot
    won[hero] += pot
    lost[hero] -= min(call, lost[hero])
    lost[villain] += pot + min(call, stacks[hero])
    uf = equity(folded, payouts, hero)
    uw = equity(won, payouts, hero)
    ul = equity(lost, payouts, hero)
    return max(0.0, min(1.0, (uf - ul) / max(1e-12, uw - ul)))
