"""Original, dependency-free five-through-seven-card Hold'em evaluator.

Cards are integers ``4 * rank + suit``: rank 0 is deuce, rank 12 is ace;
suits 0..3 are clubs, diamonds, hearts, spades.  Scores sort strongest first
with ordinary ``max``.  Category is ``score >> 20`` (0 high card, 1 pair,
2 two pair, 3 trips, 4 straight, 5 flush, 6 full house, 7 quads, 8 straight
flush).  The five lower nibbles contain descending tiebreak ranks 2..14.

The hot evaluator assumes distinct card IDs in [0, 51].  Validate external
input once with ``validate_cards`` rather than during every rollout.  All
rank-mask lookup tables are generated locally at import, with no file I/O.
"""

from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CATEGORY_NAMES = (
    "high card", "one pair", "two pair", "three of a kind", "straight",
    "flush", "full house", "four of a kind", "straight flush",
)

_BITS = tuple(1 << (card >> 2) for card in range(52))
_STRAIGHT = [0] * 8192
_TOP5 = [0] * 8192

for _mask in range(1, 8192):
    _rest = _mask
    _packed = 0
    _shift = 16
    while _rest and _shift >= 0:
        _rank = _rest.bit_length() - 1
        _packed |= (_rank + 2) << _shift
        _rest ^= 1 << _rank
        _shift -= 4
    _TOP5[_mask] = _packed
    # A-2-3-4-5 is the only straight with a low ace.
    if _mask & 0x100F == 0x100F:
        _STRAIGHT[_mask] = 5
    for _high in range(4, 13):
        if (_mask >> (_high - 4)) & 31 == 31:
            _STRAIGHT[_mask] = _high + 2

del _mask, _rest, _packed, _shift, _rank, _high


def evaluate(cards, _bits=_BITS, _straight=_STRAIGHT, _top5=_TOP5):
    """Return the best five-card score from 5, 6, or 7 validated cards."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("evaluate expects five, six, or seven cards")
    once = twice = trips = quads = 0
    suits = [0, 0, 0, 0]
    for card in cards:
        bit = _bits[card]
        quads |= trips & bit
        trips |= twice & bit
        twice |= once & bit
        once |= bit
        suits[card & 3] |= bit

    # At most one suit can have five cards when there are seven or fewer.
    flush = 0
    for suit in suits:
        if suit.bit_count() >= 5:
            flush = suit
            break
    if flush:
        straight = _straight[flush]
        if straight:
            return (8 << 20) | (straight << 16)
    if quads:
        quad = quads.bit_length() - 1
        kicker = (once ^ (1 << quad)).bit_length() + 1
        return (7 << 20) | ((quad + 2) << 16) | (kicker << 12)
    if trips:
        trip = trips.bit_length() - 1
        pair = twice ^ (1 << trip)
        if pair:
            return (6 << 20) | ((trip + 2) << 16) | ((pair.bit_length() + 1) << 12)
    if flush:
        return (5 << 20) | _top5[flush]
    straight = _straight[once]
    if straight:
        return (4 << 20) | (straight << 16)
    if trips:
        return (3 << 20) | ((trip + 2) << 16) | ((_top5[once ^ (1 << trip)] >> 4) & 0xFF00)
    if twice:
        pair = twice.bit_length() - 1
        remaining = twice ^ (1 << pair)
        if remaining:
            second = remaining.bit_length() - 1
            kicker = (once ^ (1 << pair) ^ (1 << second)).bit_length() + 1
            return (2 << 20) | ((pair + 2) << 16) | ((second + 2) << 12) | (kicker << 8)
        return (1 << 20) | ((pair + 2) << 16) | ((_top5[once ^ (1 << pair)] >> 4) & 0xFFF0)
    return _top5[once]


evaluate7 = evaluate


def evaluate5(cards):
    """Independent readable five-card reference; intentionally not the hot path."""
    if len(cards) != 5:
        raise ValueError("evaluate5 expects exactly five cards")
    ranks = sorted((card // 4 + 2 for card in cards), reverse=True)
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({card % 4 for card in cards}) == 1
    unique = sorted(counts, reverse=True)
    straight = 0
    if len(unique) == 5:
        if unique[0] - unique[-1] == 4:
            straight = unique[0]
        elif unique == [14, 5, 4, 3, 2]:
            straight = 5
    if straight and flush:
        category, kickers = 8, [straight]
    elif groups[0][0] == 4:
        category, kickers = 7, [groups[0][1], groups[1][1]]
    elif [group[0] for group in groups] == [3, 2]:
        category, kickers = 6, [groups[0][1], groups[1][1]]
    elif flush:
        category, kickers = 5, ranks
    elif straight:
        category, kickers = 4, [straight]
    elif groups[0][0] == 3:
        category, kickers = 3, [group[1] for group in groups]
    elif [group[0] for group in groups[:2]] == [2, 2]:
        category, kickers = 2, [group[1] for group in groups]
    elif groups[0][0] == 2:
        category, kickers = 1, [group[1] for group in groups]
    else:
        category, kickers = 0, ranks
    score = category << 20
    for shift, rank in zip((16, 12, 8, 4, 0), kickers):
        score |= rank << shift
    return score


def parse_card(card):
    """Parse rank/suit strings such as 'As', 'td', or '10h' into card IDs."""
    if isinstance(card, int) and not isinstance(card, bool):
        if 0 <= card < 52:
            return card
        raise ValueError("card integer outside [0, 51]")
    if not isinstance(card, str):
        raise ValueError("card must be an integer or rank/suit string")
    text = card.strip()
    if len(text) == 3 and text[:2] == "10":
        text = "T" + text[-1]
    if len(text) != 2:
        raise ValueError("card strings must contain rank then suit")
    try:
        return 4 * RANKS.index(text[0].upper()) + SUITS.index(text[1].lower())
    except ValueError:
        raise ValueError("invalid rank or suit") from None


def parse_cards(cards):
    """Parse a space-separated string or an iterable of card representations."""
    if isinstance(cards, str):
        cards = cards.split()
    return validate_cards(tuple(parse_card(card) for card in cards))


def validate_cards(cards):
    """Check uniqueness and integer encoding, retaining the supplied sequence."""
    if any(not isinstance(card, int) or isinstance(card, bool) or not 0 <= card < 52 for card in cards):
        raise ValueError("cards must be integer IDs in [0, 51]")
    if len(set(cards)) != len(cards):
        raise ValueError("duplicate cards")
    return cards


def describe(score):
    """Return (category name, nonzero tiebreak ranks) for diagnostics."""
    return CATEGORY_NAMES[score >> 20], tuple((score >> shift) & 15 for shift in (16, 12, 8, 4, 0) if (score >> shift) & 15)
