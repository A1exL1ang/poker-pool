"""Original compact Hold'em evaluator. Runtime dependencies: Python standard library.

Cards are integers rank*4+suit; ranks 0..12 represent 2..A and suits
0..3 represent clubs, diamonds, hearts, spades. Higher ranks win.
"""

RANKS = "23456789TJQKA"
SUITS = "cdhs"
CATEGORY_SHIFT = 20
CATEGORIES = ("high card", "pair", "two pair", "trips", "straight", "flush",
              "full house", "quads", "straight flush")


def parse_card(card):
    """Parse 'As', '10h', or an integer card. Reject malformed cards."""
    if isinstance(card, int):
        if 0 <= card < 52:
            return card
        raise ValueError("card integer must be in [0,51]")
    card = str(card)
    if len(card) == 3 and card[:2] == "10":
        card = "T" + card[2]
    if len(card) != 2:
        raise ValueError("card must be rank and suit")
    return RANKS.index(card[0].upper()) * 4 + SUITS.index(card[1].lower())


def card_name(card):
    return RANKS[card >> 2] + SUITS[card & 3]


def preflop_key(c1, c2=None):
    """Canonical 169-class name; accept two arguments or a pair iterable."""
    if c2 is None:
        c1, c2 = c1
    r1, r2 = c1 >> 2, c2 >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    return RANKS[r1] + RANKS[r2] + ("" if r1 == r2 else "s" if c1 & 3 == c2 & 3 else "o")


def _straight(mask):
    runs = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    if runs:
        return runs.bit_length() + 3
    return 3 if mask & 0x100F == 0x100F else -1


def _top(mask, count):
    value = 0
    for _ in range(count):
        rank = mask.bit_length() - 1
        value = (value << 4) | rank
        mask ^= 1 << rank
    return value << (4 * (5 - count))


def evaluate(cards):
    """Return an order-preserving best-five rank for 5, 6, or 7 valid cards.

    The hot path assumes distinct integer cards. Validate untrusted observations
    at their adapter boundary. Category is rank >> 20 (0..8).
    """
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    mask = 0
    for card in cards:
        rank = card >> 2
        bit = 1 << rank
        counts[rank] += 1
        suits[card & 3] |= bit
        mask |= bit
    flush_mask = 0
    for suit in suits:
        if suit.bit_count() >= 5:
            flush_mask = suit
            straight = _straight(suit)
            if straight >= 0:
                return (8 << 20) | (straight << 16)
            break
    trips = []
    pairs = []
    for rank in range(12, -1, -1):
        count = counts[rank]
        if count == 4:
            return (7 << 20) | (rank << 16) | ((mask ^ (1 << rank)).bit_length() - 1) << 12
        if count == 3:
            trips.append(rank)
        elif count == 2:
            pairs.append(rank)
    if trips and (pairs or len(trips) > 1):
        pair = max(pairs[0] if pairs else -1, trips[1] if len(trips) > 1 else -1)
        return (6 << 20) | (trips[0] << 16) | (pair << 12)
    if flush_mask:
        return (5 << 20) | _top(flush_mask, 5)
    straight = _straight(mask)
    if straight >= 0:
        return (4 << 20) | (straight << 16)
    if trips:
        return (3 << 20) | (trips[0] << 16) | (_top(mask ^ (1 << trips[0]), 2) >> 4)
    if len(pairs) >= 2:
        return (2 << 20) | (pairs[0] << 16) | (pairs[1] << 12) | (((mask ^ (1 << pairs[0]) ^ (1 << pairs[1])).bit_length() - 1) << 8)
    if pairs:
        return (1 << 20) | (pairs[0] << 16) | (_top(mask ^ (1 << pairs[0]), 3) >> 4)
    return _top(mask, 5)
