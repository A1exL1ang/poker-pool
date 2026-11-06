"""Small deterministic strategic features (no sampling in gameplay)."""


def texture_from_features(f):
    """Reuse the equity encoder's work; equivalent to texture() below."""
    early = f[0] < 5
    suited = f[3]
    fd = f[33] == 4 and f[30] < 4 and early
    nutfd = early and ((f[1] == 14 and f[31] + 1 + suited == 4) or
                       (f[2] == 14 and f[32] + 1 + suited == 4))
    sd = f[41] == 4 and f[42] < 4 and early
    return {'fd':fd,'nutfd':nutfd,'sd':sd,'draw':fd or sd,
            'overpair':bool(f[4] and f[1] > f[19]),
            'top_pair':bool((f[1] == f[19] and f[24] == 1) or (f[2] == f[19] and f[25] == 1)),
            'pair':bool(f[4] or f[24] or f[25]),'wet':f[30] >= 2 or f[42] >= 3,
            'monotone':f[30] >= 3,'board_four_flush':f[30] >= 4,
            'board_run':f[42],'ace_blocker':f[1] == 14,'two_overcards':f[2] > f[19]}


def texture(hole, board):
    ranks = [c // 4 + 2 for c in board]
    hr = [c // 4 + 2 for c in hole]
    counts = [0] * 15
    suits = [0] * 4
    for c in board:
        counts[c // 4 + 2] += 1
        suits[c % 4] += 1
    top = max(ranks)
    unique = set(ranks)
    allranks = unique | set(hr)
    if 14 in allranks:
        allranks.add(1)
    maxrun = max(sum(r in allranks for r in range(lo, lo + 5)) for lo in range(1, 11))
    boardr = set(unique)
    if 14 in boardr:
        boardr.add(1)
    boardrun = max(sum(r in boardr for r in range(lo, lo + 5)) for lo in range(1, 11))
    fs = suits.copy()
    for c in hole:
        fs[c % 4] += 1
    fd = any(fs[s] == 4 and suits[s] < 4 for s in range(4)) and len(board) < 5
    nutfd = any(fs[s] == 4 and (48 + s) in hole for s in range(4)) and len(board) < 5
    sd = maxrun == 4 and boardrun < 4 and len(board) < 5
    pocket = hr[0] == hr[1]
    overpair = pocket and hr[0] > top
    top_pair = top in hr and counts[top] == 1
    # Paired boards alone do not make hero's pocket cards a strong pair.
    pair = pocket or any(counts[r] > 0 for r in hr)
    wet = max(suits) >= 2 or boardrun >= 3
    return {'fd': fd, 'nutfd': nutfd, 'sd': sd, 'draw': fd or sd,
            'overpair': overpair, 'top_pair': top_pair, 'pair': pair,
            'wet': wet, 'monotone': max(suits) >= 3,
            'board_four_flush': max(suits) >= 4, 'board_run': boardrun,
            'ace_blocker': 14 in hr, 'two_overcards': min(hr) > top}
