"""Slow independent reference: combinations of five, rank-count sorting."""
from collections import Counter
from itertools import combinations


def five(cards):
    ranks = [c // 4 + 2 for c in cards]
    groups = sorted(((n, r) for r, n in Counter(ranks).items()), reverse=True)
    flush = len({c % 4 for c in cards}) == 1
    unique = sorted(set(ranks), reverse=True)
    straight = 0
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight = 5
    if straight and flush:
        return (8, straight)
    if groups[0][0] == 4:
        return (7, groups[0][1], groups[1][1])
    if [n for n, r in groups] == [3, 2]:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *unique)
    if straight:
        return (4, straight)
    if groups[0][0] == 3:
        return (3, groups[0][1], *sorted((r for n, r in groups[1:]), reverse=True))
    if [n for n, r in groups][:2] == [2, 2]:
        return (2, max(groups[0][1], groups[1][1]), min(groups[0][1], groups[1][1]), groups[2][1])
    if groups[0][0] == 2:
        return (1, groups[0][1], *sorted((r for n, r in groups[1:]), reverse=True))
    return (0, *unique)


def rank(cards):
    return max(five(c) for c in combinations(cards, 5))


def packed(cards):
    r = rank(cards)
    return (r[0] << 20) | sum(v << (16 - i * 4) for i, v in enumerate(r[1:]))
