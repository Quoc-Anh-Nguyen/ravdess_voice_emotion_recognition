"""Inject concise project context at supported Codex lifecycle events."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from common import additional_context, emit, load_config, project_root, read_payload, state_dir


DEV_RULES = "Repository rules: YAGNI, KISS, deterministic rendering, verify before completion."


def language_context(config: dict) -> str:
    language = config.get("language") or {}
    parts: list[str] = []
    if language.get("conversation"):
        parts.append(f"Respond to the user in {language['conversation']}.")
    if language.get("files"):
        parts.append(f"Write code and documentation in {language['files']} unless the user requests otherwise.")
    return " ".join(parts)


def compact_reminder(payload: dict, config: dict, root: Path) -> str:
    mode = config.get("cavemanMode") or {}
    if mode.get("enabled", True) is False:
        return ""
    session_id = str(payload.get("session_id") or "default")
    counter = state_dir(root) / f"tool-count-{session_id}.txt"
    try:
        count = int(counter.read_text(encoding="utf-8").strip())
    except Exception:
        return ""
    thresholds = mode.get("threshold") or {}
    red = int(thresholds.get("red", 100))
    orange = int(thresholds.get("orange", 50))
    if count >= red:
        return f"Context pressure is high after {count} local tool calls; compact at the next safe boundary and keep responses terse."
    if count >= orange:
        return f"Consider compacting at the next phase boundary ({count} local tool calls)."
    return ""


def main() -> None:
    payload = read_payload()
    event = str(payload.get("hook_event_name") or "")
    root = project_root(payload.get("cwd"))
    config = load_config(root)
    language = language_context(config)

    if event == "SessionStart":
        parts = [DEV_RULES, language, "Read AGENTS.md before repository changes."]
        last_state = state_dir(root) / "last-state.md"
        if last_state.exists():
            text = last_state.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"Previous lifecycle state: {text[:800]}")
        additional_context(event, " ".join(part for part in parts if part))
    elif event == "SubagentStart":
        additional_context(event, f"{DEV_RULES} Read the relevant AGENTS.md and verify your result before returning.")
    elif event == "UserPromptSubmit":
        message = " ".join(part for part in [language, compact_reminder(payload, config, root)] if part)
        if message:
            additional_context(event, message)
    elif event == "PostCompact":
        emit({"systemMessage": "Context was compacted. Reload AGENTS.md and preserve the active task state before editing."})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(0)
