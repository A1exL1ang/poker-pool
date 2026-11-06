"""Reproducibly build the exact evaluator asset, using only Python stdlib."""
import gzip
import itertools
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parent.parent


def score(category, ranks):
    result = category << 20
    for i, rank in enumerate(ranks[:5]):
        result |= rank << (16 - 4 * i)
    return result


def straight(ranks):
    mask = sum(1 << (r - 2) for r in set(ranks))
    for high in range(14, 5, -1):
        if mask & (31 << (high - 6)) == 31 << (high - 6):
            return high
    return 5 if mask & 0x100F == 0x100F else 0


def rank_score(ranks):
    counts = {r: ranks.count(r) for r in set(ranks)}
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    unique = sorted(counts, reverse=True)
    if groups[0][0] == 4:
        q = groups[0][1]
        return score(7, [q, max(r for r in unique if r != q)])
    trips = [r for n, r in groups if n == 3]
    pairs = [r for n, r in groups if n >= 2]
    if trips and len(pairs) >= 2:
        t = trips[0]
        return score(6, [t, max(r for r in pairs if r != t)])
    st = straight(ranks)
    if st:
        return score(4, [st])
    if trips:
        t = trips[0]
        return score(3, [t] + [r for r in unique if r != t][:2])
    if len(pairs) >= 2:
        p = sorted(pairs, reverse=True)[:2]
        return score(2, p + [max(r for r in unique if r not in p)])
    if pairs:
        p = pairs[0]
        return score(1, [p] + [r for r in unique if r != p][:3])
    return score(0, unique[:5])


def main():
    start = time.perf_counter()
    entries = []
    for size in (5, 6, 7):
        for ranks in itertools.combinations_with_replacement(range(2, 15), size):
            if any(ranks.count(r) > 4 for r in set(ranks)):
                continue
            key = sum(1 << (3 * (r - 2)) for r in ranks)
            entries.append((key, rank_score(ranks)))
    flush = []
    for mask in range(8192):
        ranks = [r + 2 for r in range(13) if mask & (1 << r)]
        st = straight(ranks) if len(ranks) >= 5 else 0
        flush.append(score(8, [st]) if st else score(5, sorted(ranks, reverse=True)[:5]))
    data = struct.pack("<I", len(entries))
    data += b"".join(struct.pack("<QI", *entry) for entry in entries)
    data += struct.pack("<8192I", *flush)
    (ROOT / "assets").mkdir(exist_ok=True)
    # Raw binary is also the input to the optional native offline equity builder.
    (ROOT / "assets" / "eval.bin").write_bytes(data)
    (ROOT / "assets" / "eval.bin.gz").write_bytes(gzip.compress(data, mtime=0))
    print(f"Generated {len(entries):,} rank multisets, 8192 flush masks in {time.perf_counter()-start:.2f}s")


if __name__ == "__main__":
    main()
