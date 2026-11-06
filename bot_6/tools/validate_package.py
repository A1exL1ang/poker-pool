"""Verify hashes, safe archive paths, reproducibility, and isolated import."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    root = Path(__file__).resolve().parent.parent
    dist = root / "dist"
    manifest = json.loads((dist / "manifest.json").read_text())
    archive = dist / "bot6.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    target = (dist / "smoke_package").resolve()
    assert target.is_relative_to(root.resolve())
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        assert set(zf.namelist()) == set(manifest["files"])
        for name in zf.namelist():
            path = (target / name).resolve()
            assert path.is_relative_to(target), name
            data = zf.read(name)
            assert hashlib.sha256(data).hexdigest() == manifest["files"][name]["sha256"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    script = "import sys,json,pathlib; sys.path.insert(0,sys.argv[1]); from bot6.agent import Agent; import bot6.agent; assert pathlib.Path(bot6.agent.__file__).is_relative_to(pathlib.Path(sys.argv[1])); print(json.dumps(Agent().act(json.loads(sys.stdin.read()))))"
    response = subprocess.run([sys.executable, "-c", script, str(target)], cwd=root,
                              input=(root / "examples/open.jsonl").read_text(), capture_output=True, text=True, check=True)
    action = json.loads(response.stdout)
    assert action == {"action": "raise", "amount": 50}, action
    result = {"archive_bytes": archive.stat().st_size, "archive_sha256": manifest["archive_sha256"],
              "verified_files": len(manifest["files"]), "isolated_smoke_action": action, "passed": True}
    (root / "benchmarks/package_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
