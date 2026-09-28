"""Count parent-turn local tool calls for compact reminders."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from common import project_root, read_payload, state_dir


def main() -> None:
    payload = read_payload()
    if payload.get("agent_id"):
        return
    session_id = str(payload.get("session_id") or "default")
    path = state_dir(project_root(payload.get("cwd"))) / f"tool-count-{session_id}.txt"
    try:
        count = int(path.read_text(encoding="utf-8").strip()) if path.exists() else 0
    except Exception:
        count = 0
    path.write_text(str(count + 1), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(0)
