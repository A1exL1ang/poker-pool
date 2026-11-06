"""Build a small local runtime bundle; does not submit, publish, or use git."""
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    root = Path.cwd().resolve()
    files = [*sorted(Path("bot3").glob("*.py")), Path("agent.py"), Path("README.md"),
             Path("assets/preflop.json"), Path("assets/pushfold.json"), Path("examples/observation.json")]
    out = Path("dist/bot_3.zip")
    out.parent.mkdir(exist_ok=True)
    manifest = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, p.as_posix())
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    verify = Path(tempfile.mkdtemp(prefix="verify_bot3_", dir="dist")).resolve()
    if not verify.is_relative_to(root):
        raise RuntimeError("verification path escaped workspace")
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:
            raise RuntimeError("corrupt archive")
        z.extractall(verify)
    code = ("import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
            "from bot3 import Agent;from bot3.interface import validate_observation,validate_action;"
            "o=json.loads((Path(sys.argv[1])/'examples/observation.json').read_text());"
            "a=Agent();print(json.dumps(validate_action(validate_observation(o),a.act(o))))")
    r = subprocess.run([sys.executable, "-I", "-c", code, str(verify)], cwd=root, capture_output=True, text=True, check=True)
    evidence = {"archive": out.as_posix(), "bytes": out.stat().st_size, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "files": manifest, "isolated_extracted_smoke_action": json.loads(r.stdout), "verification_directory": str(verify)}
    Path("evidence/package.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in evidence.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
