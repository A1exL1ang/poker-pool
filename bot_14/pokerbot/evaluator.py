"""Independent, dependency-free five-to-seven-card Hold'em hand evaluator.

Cards use ``4 * (rank - 2) + suit``; suits 0..3 have no ordering.
Scores compare directly: larger wins. The category is ``score >> 20`` with
high card=0 through straight flush=8. Invalid/duplicate input is deliberately
not checked on the hot path; use :func:`evaluate_checked` at input boundaries.

The implementation is original code. All lookup tables are generated from
the standard poker ordering at import; there are no third-party assets.
"""

from __future__ import annotations

from itertools import combinations

CATEGORY_NAMES = (
    "high_card", "one_pair", "two_pair", "three_of_a_kind", "straight",
    "flush", "full_house", "four_of_a_kind", "straight_flush",
)

_BITS = tuple(1 << (card >> 2) for card in range(52))
_SUITS = tuple(card & 3 for card in range(52))
_STRAIGHT = [0] * 8192
_TOP5 = [0] * 8192

for _mask in range(1, 8192):
    _rest = _mask
    _value = 0
    _shift = 16
    while _rest and _shift >= 0:
        _high = _rest.bit_length() - 1
        _value |= (_high + 2) << _shift
        _rest ^= 1 << _high
        _shift -= 4
    _TOP5[_mask] = _value
    _runs = _mask & (_mask >> 1) & (_mask >> 2) & (_mask >> 3) & (_mask >> 4)
    if _runs:
        _STRAIGHT[_mask] = _runs.bit_length() + 5
    elif _mask & 0x100F == 0x100F:
        _STRAIGHT[_mask] = 5

del _mask, _rest, _value, _shift, _high, _runs


def evaluate(cards, _bits=_BITS, _suits=_SUITS, _straight=_STRAIGHT, _top5=_TOP5):
    """Return the best standard-poker score from 5, 6 or 7 unique cards.

    Suits never break ties. All five kickers are encoded where relevant,
    including hands whose best five cards are entirely on the board.
    """
    a = b = c = d = 0
    for card in cards:
        bit = _bits[card]
        suit = _suits[card]
        if suit == 0:
            a |= bit
        elif suit == 1:
            b |= bit
        elif suit == 2:
            c |= bit
        else:
            d |= bit
    ranks = a | b | c | d
    # At most one suit can have five cards in a seven-card hand.
    flush = a if a.bit_count() >= 5 else b if b.bit_count() >= 5 else c if c.bit_count() >= 5 else d if d.bit_count() >= 5 else 0
    if flush:
        straight = _straight[flush]
        if straight:
            return (8 << 20) | (straight << 16)
    quads = a & b & c & d
    if quads:
        high = quads.bit_length() - 1
        return (7 << 20) | ((high + 2) << 16) | (((ranks ^ (1 << high)).bit_length() + 1) << 12)
    trips = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d)
    pairs = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d)
    if trips:
        high = trips.bit_length() - 1
        other_pairs = pairs ^ (1 << high)
        if other_pairs:
            return (6 << 20) | ((high + 2) << 16) | ((other_pairs.bit_length() + 1) << 12)
    if flush:
        return (5 << 20) | _top5[flush]
    straight = _straight[ranks]
    if straight:
        return (4 << 20) | (straight << 16)
    if trips:
        high = trips.bit_length() - 1
        return (3 << 20) | ((high + 2) << 16) | ((_top5[ranks ^ (1 << high)] >> 4) & 0xFF00)
    if pairs:
        high = pairs.bit_length() - 1
        other_pairs = pairs ^ (1 << high)
        if other_pairs:
            low = other_pairs.bit_length() - 1
            kicker = (ranks ^ (1 << high) ^ (1 << low)).bit_length() + 1
            return (2 << 20) | ((high + 2) << 16) | ((low + 2) << 12) | (kicker << 8)
        return (1 << 20) | ((high + 2) << 16) | ((_top5[ranks ^ (1 << high)] >> 4) & 0xFFF0)
    return _top5[ranks]


def evaluate_checked(cards):
    """Validate a hand before evaluating it; useful outside timed gameplay."""
    cards = tuple(cards)
    if not 5 <= len(cards) <= 7:
        raise ValueError("an evaluated hand must contain five to seven cards")
    if any(type(card) is not int or not 0 <= card < 52 for card in cards):
        raise ValueError("cards must be integer IDs in [0, 51]")
    if len(set(cards)) != len(cards):
        raise ValueError("duplicate card")
    return evaluate(cards)


def reference_evaluate(cards):
    """Slow, structurally independent combination evaluator for test evidence."""
    best = -1
    for five in combinations(cards, 5):
        ranks = [card // 4 + 2 for card in five]
        counts = {rank: ranks.count(rank) for rank in set(ranks)}
        ordered = sorted(counts, key=lambda rank: (counts[rank], rank), reverse=True)
        unique = sorted(counts)
        straight = unique[-1] if len(unique) == 5 and unique[-1] - unique[0] == 4 else 5 if unique == [2, 3, 4, 5, 14] else 0
        flush = len({card % 4 for card in five}) == 1
        pattern = sorted(counts.values(), reverse=True)
        if straight and flush:
            category, kickers = 8, [straight]
        elif pattern == [4, 1]:
            category, kickers = 7, ordered
        elif pattern == [3, 2]:
            category, kickers = 6, ordered
        elif flush:
            category, kickers = 5, sorted(ranks, reverse=True)
        elif straight:
            category, kickers = 4, [straight]
        elif pattern == [3, 1, 1]:
            category, kickers = 3, ordered
        elif pattern == [2, 2, 1]:
            category, kickers = 2, ordered
        elif pattern == [2, 1, 1, 1]:
            category, kickers = 1, ordered
        else:
            category, kickers = 0, sorted(ranks, reverse=True)
        value = category << 20
        for i, rank in enumerate(kickers):
            value |= rank << (16 - 4 * i)
        best = max(best, value)
    return best


def card_id(text: str) -> int:
    """Parse a two-character card such as ``As`` (suits c,d,h,s)."""
    if not isinstance(text, str) or len(text) != 2:
        raise ValueError("card notation must have two characters")
    try:
        return "23456789TJQKA".index(text[0].upper()) * 4 + "cdhs".index(text[1].lower())
    except ValueError:
        raise ValueError("invalid card notation") from None


def card_text(card: int) -> str:
    if type(card) is not int or not 0 <= card < 52:
        raise ValueError("card ID must be in [0,51]")
    return "23456789TJQKA"[card // 4] + "cdhs"[card % 4]
