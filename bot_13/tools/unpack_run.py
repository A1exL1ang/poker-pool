"""Validate and extract an explicitly named bot_13 result archive inside workspace."""

import argparse
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    a = p.parse_args()
    if not a.run.replace("_", "").isalnum():
        p.error("simple run name required")
    dest = ROOT / "measurements" / a.run
    dest.mkdir(exist_ok=True)
    with tarfile.open(ROOT / "measurements" / ("remote_" + a.run + ".tgz"), "r:gz") as archive:
        for member in archive.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest.resolve()) or member.issym() or member.islnk():
                raise RuntimeError("unsafe archive member")
        archive.extractall(dest)
    print("extracted", dest)


if __name__ == "__main__":
    main()
