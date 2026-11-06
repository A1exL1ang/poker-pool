"""Conservative exact river wins certified by v3 threat features.

A guarantee requires EVERY legal opponent hand to lose strictly. Shared nuts and
other no-loss-but-possible-tie hands intentionally return False. Only river hands
qualify; future cards can change a current-street maximum.
"""


def guaranteed_river_win(f):
    if len(f) != 84 or f[0] != 5:
        return False
    category, threat = f[6], f[72]
    if category > threat:
        return True
    if category < threat:
        return False
    primary = f[7]
    if category == 8:
        return primary > f[67]
    if category == 7:
        return primary > f[68]
    if category == 6:
        return (primary, f[8]) > (f[70], f[71])
    if category == 5:
        # A private card improves the flush and the hero flush is >= the maximum
        # legal opposing flush. Equality is impossible: that exact suited rank
        # is uniquely held by hero. Board-only flushes never use this shortcut.
        return bool(f[55] and f[79])
    if category == 4:
        return primary > f[64]
    if category == 3:
        return primary > f[69]
    return False
