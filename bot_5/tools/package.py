"""Build a local reviewable runtime archive; never uploads or submits it."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    assets = [ROOT / "assets" / name for name in ("flop.bin", "preflop.json", "tight_preflop.json", "ranks.bin")]
    info = {
        "provenance": "Original bot_5 source and locally generated assets; no third-party bot code or external training data.",
        "flop": {"canonical_states": 1286792, "trials_per_state": 4096, "trials_per_range": 1024, "range_widths": [.15, .35, .65, 1.0], "seed_hex": "5218394abcde9971"},
        "preflop": {"trials_per_matchup": 30000, "tight_trials_per_matchup": 100000, "tight_widths": [.03, .07]},
        "rank_states": 73775,
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in assets},
    }
    manifest = ROOT / "assets" / "manifest.json"
    manifest.write_text(json.dumps(info, indent=2) + "\n")
    sources = [ROOT / "agent.py"] + sorted((ROOT / "pokerbot").glob("*.py"))
    files = sources + assets + [manifest, ROOT / "README.md", ROOT / "examples" / "observation.json"]
    target = ROOT / "bot_5_runtime.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            z.write(path, path.relative_to(ROOT).as_posix())
    result = {"archive": target.name, "archive_bytes": target.stat().st_size, "uncompressed_bytes": sum(p.stat().st_size for p in files), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "runtime_source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "files": [p.relative_to(ROOT).as_posix() for p in files], "submitted": False}
    (ROOT / "evidence" / "package.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
