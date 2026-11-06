"""Build and smoke-test a local portable runtime archive; never submit it."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from .evidence import ROOT, fingerprint, runtime_files


def main():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    files = runtime_files() + [ROOT / "README.md", ROOT / "INTERFACE.md", ROOT / "examples" / "observation.jsonl",
                              ROOT / "benchmarks" / "RESULTS.md"]
    files += sorted((ROOT / "benchmarks").glob("*_final.json"))
    files += [ROOT / "benchmarks" / "equity_audit.json", ROOT / "benchmarks" / "exhaustive_evaluator.json",
              ROOT / "benchmarks" / "tests_final.txt"]
    target = build / "bot7.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            assert path.resolve().is_relative_to(ROOT.resolve())
            entry = zipfile.ZipInfo(str(path.relative_to(ROOT)).replace("\\", "/"), (2026, 11, 6, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    assert target.stat().st_size < 100_000_000
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=build) as directory:
        extracted = Path(directory).resolve()
        assert extracted.is_relative_to(build.resolve())
        with zipfile.ZipFile(target) as archive:
            for member in archive.namelist():
                assert (extracted / member).resolve().is_relative_to(extracted)
            archive.extractall(extracted)
        run = subprocess.run([sys.executable, "-E", "-s", str(extracted / "agent.py")],
                             input=(ROOT / "examples" / "observation.jsonl").read_text(),
                             text=True, capture_output=True, cwd=ROOT, timeout=15, check=True)
        action = json.loads(run.stdout.strip())
        assert action == {"action": "raise", "amount": 43}, action
    manifest = {"archive": "build/bot7.zip", "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "runtime_fingerprint": fingerprint(),
                "smoke_action": action, "smoke_passed": True,
                "note": "Local artifact only. Official interface compatibility unverified. No submission performed."}
    (build / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
