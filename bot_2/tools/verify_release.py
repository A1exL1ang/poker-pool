"""Source-match final reports and smoke-test the extracted runtime bundle."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
reports = [root / "benchmarks" / name for name in
           ("release_top3.json", "release_ordinal.json", "release_winner.json", "release_latency.json", "linux_latency.json")]
checks = 0
for p in reports:
    d = json.loads(p.read_text())
    for name, expected in d["gameplay_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, (p, name)
        checks += 1
bundle = root / "dist" / "bot_2_runtime.zip"
sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
smoke = root / "dist" / ("smoke_" + sha[:12])
smoke.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert (smoke / name).resolve().is_relative_to(smoke.resolve()), name
    z.extractall(smoke)
manifest = json.loads((smoke / "MANIFEST.json").read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((smoke / name).read_bytes()).hexdigest() == expected, name
result = subprocess.run([sys.executable, str(smoke / "bot.py")],
                        input=(root / "examples" / "observation.jsonl").read_text(),
                        capture_output=True, text=True, cwd=str(root), check=True)
assert not result.stderr, result.stderr
action = json.loads(result.stdout)
assert action == {"action": "raise", "amount": 50}, action
with zipfile.ZipFile(root / "dist" / "bot_2_complete.zip") as z:
    assert z.testzip() is None
    complete_manifest = json.loads(z.read("MANIFEST.json"))
    for name, expected in complete_manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == expected, name
report = {"reports_verified": len(reports), "source_hash_checks": checks,
          "runtime_bundle_bytes": bundle.stat().st_size, "runtime_sha256": sha,
          "extracted_files_verified": len(manifest),
          "complete_bundle_files_verified": len(complete_manifest), "sample_action": action,
          "passed": True}
(root / "benchmarks" / "package_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
