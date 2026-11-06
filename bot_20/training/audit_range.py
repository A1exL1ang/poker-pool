"""Independent stdlib river-label audit for selected validation cases."""
import bisect
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokerbot.evaluator import card_id, evaluate, card_text
from pokerbot.equity import preflop_percentile


def audit(hole_text, board_text, n, fraction, pressure, samples=10000):
    hole = tuple(map(card_id, hole_text.split()))
    board = tuple(map(card_id, board_text.split()))
    legal = [c for c in range(52) if c not in board]
    combos = [(a, b) for i, a in enumerate(legal) for b in legal[i + 1:]]
    ranks = [evaluate((*h, *board)) for h in combos]
    ordered = sorted(ranks)
    weighted, cdf = [], []
    total = 0
    for h, value in zip(combos, ranks):
        if h[0] in hole or h[1] in hole:
            continue
        p = preflop_percentile(h)
        weight = 1 if fraction >= .999 or p <= fraction else .02 * math.exp(-6 * (p - fraction))
        hs = (bisect.bisect_left(ordered, value) + bisect.bisect_right(ordered, value)) / (2 * len(ordered))
        weight *= 1 - pressure + pressure * (.02 + 3 * hs ** 6)
        total += weight
        cdf.append(total)
        weighted.append((h, value, weight))
    hero_value = evaluate((*hole, *board))
    rng = random.Random(732190)
    shares = []
    example_losses = []
    for _ in range(samples):
        used = set(hole + board)
        hands, loss, ties = [], False, 1
        for j in range(n):
            for attempt in range(1000):
                h, value, weight = weighted[bisect.bisect_left(cdf, rng.random() * total)]
                if h[0] not in used and h[1] not in used:
                    break
            else:
                remaining = [item for item in weighted if item[0][0] not in used and item[0][1] not in used]
                target = rng.random() * sum(item[2] for item in remaining)
                for h, value, weight in remaining:
                    target -= weight
                    if target <= 0:
                        break
            used.update(h)
            hands.append((h, value))
            loss |= value > hero_value
            ties += value == hero_value
        shares.append(0 if loss else 1 / ties)
        if loss and len(example_losses) < 3:
            example_losses.append([" ".join(map(card_text, h)) for h, v in hands if v > hero_value])
    mean = sum(shares) / samples
    return {"hole": hole_text, "board": board_text, "opponents": n, "range_fraction": fraction, "pressure": pressure,
            "samples": samples, "equity": mean, "stderr": math.sqrt(sum((v - mean) ** 2 for v in shares) / samples ** 2),
            "first_opponent_loses_to_hero_probability": sum(w for h, v, w in weighted if v < hero_value) / total,
            "example_winning_opponent_hands": example_losses}


if __name__ == "__main__":
    results = [
        audit("9s 3s", "8c 9d 3c 3h Td", 7, .07374914735555649, .8887242078781128),
        audit("Ac Ad", "3c As 4c 5h 7d", 5, .13715794682502747, 1),
        audit("Th Ts", "6d 5s Ac Td 7s", 5, .2255862057209015, 1),
    ]
    Path("measurements/equity_range_audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
