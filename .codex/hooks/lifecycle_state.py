"""Persist lightweight Codex lifecycle state without parsing transcript internals."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from common import load_config, project_root, read_payload, state_dir


def purge_old_files(directory: Path, max_age_days: int) -> None:
    cutoff = time.time() - max(max_age_days, 1) * 86400
    for path in directory.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def main() -> None:
    payload = read_payload()
    event = str(payload.get("hook_event_name") or "")
    root = project_root(payload.get("cwd"))
    directory = state_dir(root)
    config = load_config(root)

    if event == "PreCompact":
        purge_old_files(directory, int(config.get("compactDay", 3)))
        session_id = str(payload.get("session_id") or "default")
        (directory / f"tool-count-{session_id}.txt").unlink(missing_ok=True)
        with (directory / "compaction-log.txt").open("a", encoding="utf-8") as stream:
            stream.write(f"{int(time.time())}\t{payload.get('trigger', 'unknown')}\n")
        return

    if event in {"Stop", "SubagentStop"}:
        state = {
            "event": event,
            "session_id": payload.get("session_id"),
            "turn_id": payload.get("turn_id"),
            "cwd": str(root),
            "transcript_path": payload.get("transcript_path"),
            "saved_at": int(time.time()),
        }
        (directory / "last-state.md").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(0)
