"""Compact, read-only status for the orchestrator; never passed to workers."""
import datetime as dt
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
control = root / "orchestration"
state = json.loads((control / "status.json").read_text(encoding="utf-8"))
result = {k: state.get(k) for k in ("status", "started_at", "last_poll_at", "max_concurrent", "service_tier")}
result["workers"] = []
for worker in state["workers"]:
    row = {k: worker.get(k) for k in ("number", "model", "reasoning", "status", "pid", "deadline", "exit_code", "export_path", "export_error")}
    log = control / "logs" / f"bot_{worker['number']}.jsonl"
    events = []
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    row["events"] = len(events)
    row["thread_id"] = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), None)
    messages = [e["item"].get("text", "") for e in events if e.get("item", {}).get("type") == "agent_message"]
    row["last_message"] = messages[-1][-1000:] if messages else None
    failures = [e for e in events if e.get("type") in ("error", "turn.failed")]
    row["errors"] = failures[-2:]
    workspace = Path(worker["workspace"])
    progress = workspace / ".control" / "STATUS.txt"
    row["progress"] = progress.read_text(encoding="utf-8", errors="replace")[-2000:] if progress.exists() else None
    row["files"] = sorted(p.name for p in workspace.iterdir() if not p.name.startswith("."))
    result["workers"].append(row)
print(json.dumps(result, indent=2))
