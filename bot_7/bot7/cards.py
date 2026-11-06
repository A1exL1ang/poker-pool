"""Cards are rank-major integers: 0=2c, 1=2d, ..., 51=As."""

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


def card(value):
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 52:
        return value
    if isinstance(value, str):
        text = value.strip().replace("10", "T")
        if len(text) == 2:
            return RANK_CHARS.index(text[0].upper()) * 4 + SUIT_CHARS.index(text[1].lower())
    raise ValueError("card must be 0..51 or a string such as As or Td")


def name(value):
    return RANK_CHARS[value // 4] + SUIT_CHARS[value % 4]


def hand_class(a, b):
    hi, lo = max(a // 4, b // 4), min(a // 4, b // 4)
    if hi == lo:
        return RANK_CHARS[hi] * 2
    return RANK_CHARS[hi] + RANK_CHARS[lo] + ("s" if a % 4 == b % 4 else "o")

