"""Bounded range-conditioned sampling with reusable board evaluation.

No network, files, or unbounded searches occur during a decision. Sampling is
deterministic for a public state plus the Agent seed. Preflop lookup is O(1).
"""

import json
from pathlib import Path
from .cards import class_index, board_state, evaluate_two, BITS, COUNT, STRAIGHT

with (Path(__file__).resolve().parent.parent / "assets" / "preflop.json").open() as _f:
    TABLE = json.load(_f)
CUTOFFS = tuple(TABLE["cutoffs"])
PERCENTILE = tuple(TABLE["percentile"])
PREFLOP = tuple(TABLE["versus_range"])
MULTIWAY = tuple(TABLE["multiway"])
COMBOS = tuple((a, b) for a in range(52) for b in range(a + 1, 52))
RANGES = tuple(tuple((a, b) for a, b in COMBOS if PERCENTILE[class_index(a, b)] <= cut)
               for cut in CUTOFFS)


def range_index(width):
    return min(range(len(CUTOFFS)), key=lambda i: abs(CUTOFFS[i] - width))


def preflop_equity(a, b, width=1.0, opponents=1):
    h = class_index(a, b)
    eq = PREFLOP[h * 10 + range_index(width)]
    if opponents > 1:
        # Calibrate the multiway penalty with offline joint-deal simulations.
        indexes = (0, 0, 1, 2, 3, 3, 4, 4)
        random_hu = MULTIWAY[h * 5]
        actual = MULTIWAY[h * 5 + indexes[min(opponents, 7)]]
        eq *= actual / max(random_hu, .01)
    return eq


def _likelihood(score, state, a, b, pressure, street):
    """Soft likelihood of the observed postflop action; never zero support."""
    if pressure <= 0:
        return 1.0
    cat = score >> 20
    board_pairs = state[1]
    # A paired board is not private hand strength.
    if cat == 1 and board_pairs and not ((BITS[a] | BITS[b]) & board_pairs):
        cat = 0
    if cat >= 3:
        value = 1.0
    elif cat == 2:
        value = .85
    elif cat == 1:
        pair = (score >> 16) & 15
        high_board = state[0].bit_length() + 1
        value = .68 if pair >= high_board else .35
    else:
        value = .10
    if street < 3:
        mask = state[0] | BITS[a] | BITS[b]
        # Four-flush or four-to-straight gives semi-bluffs realistic weight.
        suits = list(state[4:])
        suits[a & 3] |= BITS[a]
        suits[b & 3] |= BITS[b]
        draw = any(COUNT[s] >= 4 for s in suits)
        if not draw:
            for shift in range(9):
                if COUNT[(mask >> shift) & 31] >= 4:
                    draw = True
                    break
        if draw:
            value = max(value, .48)
    return (1.0 - pressure) + pressure * value


def postflop_equity(hole, board, width, opponents, pressure, seed,
                    runouts=6, samples=8):
    """Returns (multiway showdown equity, heads-up equity, current hand score).

    Conditional-on-runout powers approximate multiple opponents. This ignores
    mutual opponent card removal; ties are divided among estimated co-winners.
    """
    a, b = hole
    current = board_state(board)
    made = evaluate_two(current, a, b)
    nmissing = 5 - len(board)
    known = (1 << a) | (1 << b)
    for c in board:
        known |= 1 << c
    pool = RANGES[range_index(width)]
    plen = len(pool)
    # Xorshift32: private per-call state; deterministic, no Python RNG overhead.
    x = (seed & 0xFFFFFFFF) or 1
    total = hu_total = 0.0
    if not nmissing:
        runouts, samples = 1, runouts * samples
    for _ in range(runouts):
        used = known
        extra = []
        for k in range(nmissing):
            # A bounded fallback scan prevents adversarial RNG rejection loops.
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= x >> 17
            x ^= (x << 5) & 0xFFFFFFFF
            c = x % 52
            while used & (1 << c):
                c = (c + 1) % 52
            extra.append(c)
            used |= 1 << c
        state = board_state(board + tuple(extra)) if nmissing else current
        hero_score = evaluate_two(state, a, b)
        win = tie = weight = 0.0
        for _j in range(samples):
            # Sample from preflop range; bounded ten attempts, deterministic scan.
            for attempt in range(10):
                x ^= (x << 13) & 0xFFFFFFFF
                x ^= x >> 17
                x ^= (x << 5) & 0xFFFFFFFF
                u, v = pool[x % plen]
                if not used & ((1 << u) | (1 << v)):
                    break
            else:
                # Only reached for exceptionally tight blocked ranges.
                for u, v in pool:
                    if not used & ((1 << u) | (1 << v)):
                        break
                else:
                    u = next(c for c in range(52) if not used & (1 << c))
                    v = next(c for c in range(u + 1, 52) if not used & (1 << c))
            villain_score = evaluate_two(state, u, v)
            w = 1.0
            if pressure > 0:
                now = evaluate_two(current, u, v) if nmissing else villain_score
                w = _likelihood(now, current, u, v, pressure, len(board) - 2)
            weight += w
            if hero_score > villain_score:
                win += w
            elif hero_score == villain_score:
                tie += w
        pwin, ptie = win / weight, tie / weight
        hu_total += pwin + .5 * ptie
        if opponents == 1:
            total += pwin + .5 * ptie
        elif ptie < 1e-9:
            total += pwin ** opponents
        else:
            # Integral of (pwin + t*ptie)^n over t accounts for split pots.
            total += ((pwin + ptie) ** (opponents + 1) - pwin ** (opponents + 1)) / ((opponents + 1) * ptie)
    return total / runouts, hu_total / runouts, made


def texture(board):
    state = board_state(board)
    suits = max(COUNT[s] for s in state[4:])
    mask = state[0]
    connected = max(COUNT[(mask >> k) & 31] for k in range(9))
    wet = min(1.0, max(0.0, (suits - 1) * .24 + max(0, connected - 2) * .23))
    return wet, state
