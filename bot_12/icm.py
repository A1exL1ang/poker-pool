"""Exact, small-table Independent Chip Model; no assumed contest payouts."""
from functools import lru_cache


@lru_cache(maxsize=4096)
def _hero_equity(stacks, payouts, hero):
    h = stacks[hero]
    alive = sum(s > 0 for s in stacks)
    if h <= 0:
        return payouts[min(alive, len(payouts) - 1)] if payouts else 0.0
    opponents = tuple(s for i, s in enumerate(stacks) if i != hero and s > 0)
    n = len(opponents)
    total = h + sum(opponents)
    if not payouts:
        return h / total
    prizes = payouts[:n+1]
    # Plackett-Luce is equivalent to independent exponential finish times.
    # E[rank] = 1 + sum_j P(j before hero), so linear payouts need no DP.
    step = (prizes[0]-prizes[-1])/n if n else 0.0
    if n == 0 or all(abs(prizes[k]-(prizes[0]-k*step)) < 1e-12 for k in range(n+1)):
        return prizes[0] - step*sum(s/(h+s) for s in opponents)
    # Probability of each unordered set having finished ahead of hero.
    size = 1 << n
    probability = [0.0] * size
    removed = [0] * size
    probability[0] = 1.0
    answer = 0.0
    for mask in range(size):
        if mask:
            bit = mask & -mask
            removed[mask] = removed[mask ^ bit] + opponents[bit.bit_length() - 1]
        p = probability[mask]
        if not p:
            continue
        denom = total - removed[mask]
        rank = mask.bit_count()
        prize = payouts[rank] if rank < len(payouts) else 0.0
        answer += p * h / denom * prize
        if rank + 1 >= len(payouts) or not any(payouts[rank + 1:]):
            continue
        for j, stack in enumerate(opponents):
            if not mask & (1 << j):
                probability[mask | (1 << j)] += p * stack / denom
    return answer


def hero_equity(stacks, payouts, hero=0):
    """Expected points for hero. Zero stacks already occupy trailing places.

    Payouts are ordered first to last. Missing places pay zero. For a zero
    hero stack, use the next eliminated place (ties require external handling).
    """
    prizes = tuple(payouts) + (0.0,) * max(0, len(stacks)-len(payouts))
    return _hero_equity(tuple(stacks), prizes, hero)


def call_threshold(obs, opponent, cost, eligible_pot):
    """Exact ICM threshold for a heads-up, single-pot call, else chip odds.

    Fold baseline awards the existing pot to the remaining opponent. The
    caller must supply only a heads-up single-pot situation. Folded players'
    dead chips are included. Win/loss outcomes conserve all table chips.
    """
    odds = cost / max(1, eligible_pot + cost)
    payouts = obs.get("payouts")
    if not payouts or not cost:
        return odds
    hero = obs["hero"]
    stacks = list(obs["stacks"])
    pot = obs["pot"]
    folded = stacks.copy()
    folded[opponent] += pot
    win = stacks.copy()
    win[hero] += eligible_pot
    win[opponent] += pot - eligible_pot
    lose = stacks.copy()
    lose[hero] -= cost
    lose[opponent] += pot + cost
    uf = hero_equity(folded, payouts, hero)
    uw = hero_equity(win, payouts, hero)
    ul = hero_equity(lose, payouts, hero)
    if uw <= ul:
        return 1.0
    return min(1.0, max(odds, (uf - ul) / (uw - ul)))
