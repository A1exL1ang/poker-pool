"""Verify the crafted quad-board/random feature audit written by native_validate."""
import hashlib
import json
from pathlib import Path
import struct

from bot.equity import features


def main():
    data = Path('artifacts/equity_feature_audit.bin').read_bytes()
    row = struct.Struct('<41f7i')
    assert len(data) % row.size == 0
    mismatches = 0
    for offset in range(0, len(data), row.size):
        values = row.unpack_from(data, offset)
        cards = [c for c in values[41:] if c >= 0]
        mismatches += list(values[:41]) != features(cards[:2], cards[2:])
    report = dict(rows=len(data) // row.size, mismatches=mismatches,
                  source_sha256=hashlib.sha256(data).hexdigest(),
                  cases='All 13 quads turn boards x1128 legal hole pairs, plus5000 random flop/turn/river cases')
    Path('artifacts/feature_parity.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert mismatches == 0


if __name__ == '__main__':
    main()
