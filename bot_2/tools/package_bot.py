"""Build a small local runtime bundle. Does not contact any submission service."""
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    files = sorted([Path("bot.py"), Path("README.md"), Path("INTERFACE.md"), Path("requirements.txt"),
                    Path("examples/observation.jsonl"),
                    Path("assets/preflop.json"), Path("assets/equity_model.json"),
                    *Path("pokerbot").glob("*.py")])
    manifest = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    Path("dist").mkdir(exist_ok=True)
    bundle = Path("dist/bot_2_runtime.zip")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            info = zipfile.ZipInfo(p.as_posix(), date_time=(2026, 11, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
        info = zipfile.ZipInfo("MANIFEST.json", date_time=(2026, 11, 6, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, json.dumps(manifest, indent=2) + "\n")
    result = {"file": bundle.as_posix(), "bytes": bundle.stat().st_size,
              "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
              "uncompressed_source_bytes": sum(p.stat().st_size for p in files),
              "files": manifest}
    Path("dist/manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    complete_files = sorted(set(files + [Path("RULES.txt"),
        *Path("tools").glob("*.py"), *Path("tools").glob("*.cpp"),
        *Path("tools").glob("*.sbatch"), *Path("tests").glob("*.py"),
        *Path("assets").glob("*.jsonl"), *Path("benchmarks").rglob("*.json"),
        *Path("benchmarks").rglob("*.txt"), *Path("benchmarks").rglob("*.log")]))
    complete = Path("dist/bot_2_complete.zip")
    complete_manifest = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in complete_files}
    with zipfile.ZipFile(complete, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in complete_files:
            info = zipfile.ZipInfo(p.as_posix(), date_time=(2026, 11, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
        info = zipfile.ZipInfo("MANIFEST.json", date_time=(2026, 11, 6, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, json.dumps(complete_manifest, indent=2) + "\n")
    result["complete_bundle"] = {"file": complete.as_posix(), "bytes": complete.stat().st_size,
                                  "sha256": hashlib.sha256(complete.read_bytes()).hexdigest()}
    assert bundle.stat().st_size < 100_000_000 and complete.stat().st_size < 100_000_000
    Path("dist/manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
