"""Collect source-matched final evidence after tests, benchmarks and packaging."""
import hashlib
import json
import re
import time
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def check_fingerprints(report):
    for name, expected in report["sha256"].items():
        p = Path(name.replace("\\", "/"))
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("stale final evidence: " + str(p))


def tests(p):
    s = Path(p).read_text(encoding="utf-8-sig")
    if not re.search(r"^OK(?: \(skipped=\d+\))?\s*$", s, re.M):
        raise RuntimeError("test suite did not pass: " + p)
    return {"ran": int(re.search(r"Ran (\d+) tests", s).group(1)),
            "skipped": int(m.group(1)) if (m := re.search(r"OK \(skipped=(\d+)\)", s)) else 0,
            "log": p}


def main():
    windows = read("evidence/final_windows_latency.json")
    linux = read("evidence/final_linux_latency.json")
    tournament = read("evidence/final_deadline_mixed_800.json")
    for report in (windows, linux, tournament):
        check_fingerprints(report)
    package = read("evidence/package.json")
    archive = Path(package["archive"])
    if hashlib.sha256(archive.read_bytes()).hexdigest() != package["sha256"]:
        raise RuntimeError("package fingerprint mismatch")
    for p, expected in package["files"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != expected:
            raise RuntimeError("stale packaged file: " + p)
    result = {"status": "complete", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "bot": "bot_3", "interface": "local-v1; official adapter not supplied",
              "windows_tests": tests("evidence/tests_final.txt"), "linux_tests": tests("evidence/tests_linux_final.txt"),
              "exhaustive_five_card_audit": read("evidence/exhaustive_five_card_audit.json"),
              "equity_validation": read("evidence/model_validation.json"),
              "windows_latency": windows["latency"], "linux_latency": linux["latency"],
              "deadline_enforced_tournament": tournament["tournaments"],
              "package": package, "gameplay_sha256": windows["sha256"], "remote_jobs_launched": [],
              "validation_sources_sha256": {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in [*Path("tools").glob("*.py"), *Path("tools").glob("*.cpp"), *Path("tests").glob("*.py"), Path("RULES.txt"), Path("README.md")]},
              "limitations": ["Official Agent interface, payouts and later blind levels absent.",
                              "Heuristic full-game policy and modeled opponent ranges; multiway equity approximate.",
                              "Measured wall-clock calls sometimes exceed 1 ms; hard deadline not guaranteed.",
                              "Synthetic opponents and assumed blind doubling; no official-engine or actual-entrant validation."]}
    Path("evidence/FINAL.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "utc": result["utc"], "archive": str(archive), "archive_bytes": archive.stat().st_size,
                      "tests": result["windows_tests"], "linux_tests": result["linux_tests"],
                      "deadline_enforced_win_rate": tournament["tournaments"]["mean_terminal_chip_share"]}, indent=2))


if __name__ == "__main__":
    main()
