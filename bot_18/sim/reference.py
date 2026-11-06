"""Slow, independent best-of-five evaluator used as a simulator fallback."""
from collections import Counter
from itertools import combinations


def five(cards):
    ranks = sorted((c // 4 + 2 for c in cards), reverse=True)
    counts = Counter(ranks)
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    unique = sorted(counts, reverse=True)
    straight = 0
    if len(unique) == 5:
        if unique[0] - unique[-1] == 4:
            straight = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight = 5
    if flush and straight:
        value = (8, straight)
    elif groups[0][0] == 4:
        value = (7, groups[0][1], groups[1][1])
    elif [g[0] for g in groups] == [3, 2]:
        value = (6, groups[0][1], groups[1][1])
    elif flush:
        value = (5, *ranks)
    elif straight:
        value = (4, straight)
    elif groups[0][0] == 3:
        value = (3, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    elif [g[0] for g in groups[:2]] == [2, 2]:
        pairs = sorted((g[1] for g in groups[:2]), reverse=True)
        value = (2, *pairs, groups[2][1])
    elif groups[0][0] == 2:
        value = (1, groups[0][1], *sorted((r for r in ranks if r != groups[0][1]), reverse=True))
    else:
        value = (0, *ranks)
    return value + (0,) * (6 - len(value))


def evaluate(cards):
    if not 5 <= len(cards) <= 7 or len(set(cards)) != len(cards):
        raise ValueError("expected five to seven unique cards")
    return max(five(c) for c in combinations(cards, 5))

