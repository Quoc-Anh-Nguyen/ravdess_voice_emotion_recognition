"""Shared helpers for repository-local Codex hooks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def read_payload() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def project_root(cwd: str | None = None) -> Path:
    start = Path(cwd or os.getcwd()).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=start,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip()).resolve()
    except Exception:
        pass

    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def load_config(root: Path) -> dict[str, Any]:
    path = root / ".ck.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def state_dir(root: Path) -> Path:
    path = root / ".codex" / "session-data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def emit(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False))


def additional_context(event: str, message: str) -> None:
    emit({
        "hookSpecificOutput": {
            "hookEventName": event,
            "additionalContext": message,
        }
    })


_PATCH_PATH_RE = re.compile(r"^\*\*\* (?:Add|Update|Delete) File:\s*(.+?)\s*$", re.MULTILINE)


def touched_paths(payload: dict[str, Any]) -> list[Path]:
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return []

    raw_paths: list[str] = []
    for key in ("file_path", "path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            raw_paths.append(value.strip())

    command = tool_input.get("command")
    if isinstance(command, str):
        raw_paths.extend(match.strip() for match in _PATCH_PATH_RE.findall(command))

    root = project_root(payload.get("cwd"))
    result: list[Path] = []
    seen: set[str] = set()
    for raw in raw_paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        normalized = str(candidate.resolve(strict=False)).lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(candidate)
    return result
