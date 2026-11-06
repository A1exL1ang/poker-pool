"""Unpack only this bot's fetched archive; validate every extraction target."""

from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "measurements" / "train_v1"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ROOT / "measurements" / "remote_train_v1.tgz", "r:gz") as archive:
        for member in archive.getmembers():
            target = (DEST / member.name).resolve()
            if not target.is_relative_to(DEST.resolve()) or member.issym() or member.islnk():
                raise RuntimeError("unsafe archive member")
        archive.extractall(DEST)
    shutil.copy2(DEST / "src" / "assets" / "preflop.json", ROOT / "assets" / "preflop.json")
    # This first job predates the private-draw audit fix. Retain its Linux library
    # with its source checkpoint until the release build is compiled separately.
    print("unpacked", DEST, "and installed the 200k/cell preflop table")


if __name__ == "__main__":
    main()
