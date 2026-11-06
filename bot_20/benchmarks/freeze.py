"""Freeze local source/assets before final evaluation; no publishing/submission."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [root / "agent.py", root / "benchmarks" / "protocol.json"]
    for directory in ("pokerbot", "simulator", "benchmarks", "tests"):
        paths.extend((root / directory).rglob("*.py"))
    for pattern in ("*.py", "*.cpp", "*.sh"):
        paths.extend((root / "training").glob(pattern))
    paths.extend(p for p in (root / "pokerbot" / "assets").iterdir() if p.is_file())
    files = {p.relative_to(root).as_posix(): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                           "bytes": p.stat().st_size} for p in sorted(set(paths))}
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "files": files,
              "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
              "purpose": "Candidate/source freeze before independently seeded final local evaluation"}
    path = root / "artifacts" / "final_manifest.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"file": str(path), "fingerprint": result["fingerprint"], "count": len(files)}))


if __name__ == "__main__":
    main()
