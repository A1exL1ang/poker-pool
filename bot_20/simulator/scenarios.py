"""Explicitly HYPOTHETICAL scoring/speed scenarios for sensitivity tests.

Only the initial 10/20 blinds, 25-hand level period, 2000 starting chips, and
200-hand limit come from RULES.txt. Later blinds and every payout here are local
stress assumptions, not contest information. All payout vectors sum to one.
"""

BLINDS = {
    "slow": [(10, 20), (15, 30), (20, 40), (30, 60), (50, 100), (75, 150), (100, 200), (150, 300)],
    "medium": [(10, 20), (15, 30), (25, 50), (50, 100), (75, 150), (100, 200), (150, 300), (200, 400)],
    "fast": [(10, 20), (20, 40), (40, 80), (80, 160), (160, 320), (320, 640), (640, 1280), (1280, 2560)],
}

PAYOUTS = {
    "top3": [.5, .3, .2, 0., 0., 0., 0., 0.],
    "winner": [1., 0., 0., 0., 0., 0., 0., 0.],
    "flat3": [1 / 3, 1 / 3, 1 / 3, 0., 0., 0., 0., 0.],
    "rank": [i / 36 for i in range(8, 0, -1)],
}

FIELDS = {
    "mixed": ["tag", "nit", "station", "maniac", "equity", "tag", "station"],
    "tight": ["tag", "nit", "tag", "nit", "tag", "equity", "tag"],
    "loose": ["station", "maniac", "station", "random", "maniac", "equity", "station"],
    "passive": ["station", "station", "nit", "equity", "station", "nit", "station"],
    "robust": ["tag", "model_equity", "agent:Agent", "agent:Agent", "station", "maniac", "equity"],
}
