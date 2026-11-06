"""Local observation validation and legal action normalization.

The official starter interface is unavailable. This is an explicit local API,
not heuristic guessing of unseen contest observation field names.
"""


def safe_action(obs):
    return {"action": "check" if obs.get("to_call", 0) <= 0 else "fold", "amount": 0}


def legalize(obs, action):
    hero = obs["hero"]
    stack = max(0, int(obs["stacks"][hero]))
    cost = min(stack, max(0, int(obs["to_call"])))
    kind = action.get("action", "fold")
    if kind == "raise" and obs.get("can_raise", False):
        current = int(obs["street_bets"][hero])
        maximum = min(current + stack, int(obs["max_raise_to"]))
        minimum = int(obs["min_raise_to"])
        requested = int(action.get("amount", 0))
        target = min(maximum, max(requested, minimum))
        if target > current + cost and (target >= minimum or target == maximum):
            return {"action": "raise", "amount": target}
    if kind in ("call", "raise") and cost:
        return {"action": "call", "amount": cost}
    return safe_action(obs)


def validate_observation(o):
    """Strict integration/debug check, intentionally outside timed act()."""
    n = len(o["stacks"])
    assert 2 <= n <= 8 and 0 <= o["hero"] < n and 0 <= o["button"] < n
    assert len(o["hole"]) == 2 and len(o["board"]) in (0, 3, 4, 5)
    assert len(o["board"]) == (0, 3, 4, 5)[o["street"]]
    cards = list(o["hole"]) + list(o["board"])
    assert len(set(cards)) == len(cards) and all(type(c) is int and 0 <= c < 52 for c in cards)
    for key in ("stacks", "street_bets", "contributions", "in_hand"):
        assert len(o[key]) == n
    for key in ("stacks", "street_bets", "contributions"):
        assert all(type(x) is int and x >= 0 for x in o[key])
    assert all(a <= b for a, b in zip(o["street_bets"], o["contributions"]))
    assert o["pot"] == sum(o["contributions"])
    assert o["to_call"] >= 0 and o["big_blind"] > 0
    assert o["max_raise_to"] == o["street_bets"][o["hero"]] + o["stacks"][o["hero"]]
    if "payouts" in o and o["payouts"] is not None:
        assert len(o["payouts"]) >= n
        assert all(a >= b >= 0 for a, b in zip(o["payouts"], o["payouts"][1:]))
    return True
