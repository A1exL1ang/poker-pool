"""Rebuild original rank evaluator asset; standard library only."""
from itertools import combinations_with_replacement
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pokerbot.cards import _evaluate_direct


def main():
    rows = {}
    for n in (5, 6, 7):
        for ranks in combinations_with_replacement(range(13), n):
            if any(ranks.count(r) > 4 for r in set(ranks)):
                continue
            cards = [(r << 2) | (i & 3) for i, r in enumerate(ranks)]
            key = sum(1 << (r * 3) for r in ranks)
            rows[key] = _evaluate_direct(cards)
    path = Path(__file__).resolve().parent.parent / "assets" / "ranks.bin"
    with path.open("wb") as f:
        f.write(b"B5RANK01")
        for key, value in sorted(rows.items()):
            f.write(struct.pack("<QI", key, value))
    print(f"wrote {len(rows)} rank states; {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
