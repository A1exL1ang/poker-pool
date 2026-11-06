"""Save this bot's source checkpoint; explicitly exclude metadata and unrelated files."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
DIRECTORIES = ("bot13", "native", "lab", "tools", "tests", "assets", "plans")
EXTENSIONS = {".py", ".cpp", ".h", ".json", ".md", ".sh", ".dll", ".so"}


def snapshot(path):
    files = [ROOT / "agent.py", ROOT / "RULES.txt"]
    if (ROOT / "README.md").exists():
        files.append(ROOT / "README.md")
    for directory in DIRECTORIES:
        base = ROOT / directory
        if base.exists():
            files += [p for p in base.rglob("*") if p.is_file() and p.suffix in EXTENSIONS and "__pycache__" not in p.parts and not p.is_symlink()]
    path.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            relative = f.relative_to(ROOT).as_posix()
            hashes[relative] = hashlib.sha256(f.read_bytes()).hexdigest()
            z.write(f, relative)
        z.writestr("CHECKPOINT.json", json.dumps(hashes, indent=2))
    print(path, path.stat().st_size, "bytes", len(hashes), "files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    snapshot(p.parse_args().output)
