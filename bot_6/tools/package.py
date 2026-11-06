"""Build a reproducible runtime ZIP and SHA-256 manifest; never submit it."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent.parent
    files = [root / "agent.py", root / "README.md", root / "requirements.txt", root / "assets/preflop.json"]
    files += sorted((root / "bot6").glob("*.py"))
    files += sorted((root / "examples").glob("*.jsonl"))
    target = root / "dist"
    target.mkdir(exist_ok=True)
    manifest = {}
    archive = target / "bot6.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(files):
            data = path.read_bytes()
            name = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 6, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, data, compresslevel=9)
            manifest[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    output = {"archive": archive.name, "archive_bytes": archive.stat().st_size,
              "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "runtime_source_and_asset_bytes": sum(v["bytes"] for v in manifest.values()), "files": manifest,
              "notice": "Local artifact only, not submitted or officially interface-compatible."}
    (target / "manifest.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({k: v for k, v in output.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
