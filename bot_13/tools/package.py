"""Create a self-contained local gameplay package, excluding all control/compute data."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "dist" / "bot_13.zip")
    a = p.parse_args()
    relative = ["agent.py", "README.md", "RULES.txt", "native/bot13.cpp", "native/build.py", "native/README.md"]
    relative += [f.relative_to(ROOT).as_posix() for f in (ROOT / "bot13").glob("*.py")]
    relative += ["assets/preflop.json", "assets/pushfold.json", "assets/policy.json"]
    for binary in ("native/bot13.dll", "native/libbot13.so"):
        if (ROOT / binary).is_file():
            relative.append(binary)
    files = {}
    for name in sorted(set(relative)):
        path = ROOT / name
        if path.is_file():
            files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    if not {"native/bot13.dll", "native/libbot13.so"} <= files.keys():
        raise SystemExit("Build and verify both intended x86-64 platform libraries before final packaging.")
    raw_size = sum(f["bytes"] for f in files.values())
    if raw_size >= 100_000_000:
        raise SystemExit("Uncompressed package exceeds the stated 100MB limit.")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "entrypoint": "agent:Agent", "interface": "independent local canonical; official adapter unverified",
                "network_during_play": False, "uncompressed_bytes": raw_size, "files": files}
    with zipfile.ZipFile(a.output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:
            z.write(ROOT / name, name)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    (a.output.parent / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(a.output, a.output.stat().st_size, "compressed bytes;", raw_size, "uncompressed bytes")


if __name__ == "__main__":
    main()
