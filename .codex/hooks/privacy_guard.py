"""Block local tool calls that directly access sensitive files."""

from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from common import emit, load_config, project_root, read_payload, touched_paths


SENSITIVE_NAMES = (
    ".env", ".env.*", "*.key", "*.pem", "*.pfx", "*.p12", "*.keystore",
    "*.jks", "*credential*", "*secret*", "*password*", "*passwd*", "*.cert",
    "id_rsa", "id_ecdsa", "id_ed25519", "id_dsa",
)
SENSITIVE_PARTS = ("/.ssh/", "/secrets/", "/credentials/", "/.gnupg/")
COMMAND_PATTERN = re.compile(
    r"(?:^|[\\/\s'\"])(\.env(?:\.[\w.-]+)?|[^\s'\"]*\.(?:key|pem|pfx|p12)|"
    r"id_(?:rsa|ecdsa|ed25519|dsa)|[^\s'\"]*(?:credential|secret|password|passwd)[^\s'\"]*)",
    re.IGNORECASE,
)


def allow_list(config: dict) -> set[str]:
    values = (config.get("privacyBlock") or {}).get("allowList") or []
    return {str(value).lower().replace("\\", "/") for value in values}


def path_match(path: Path, allowed: set[str]) -> str | None:
    normalized = str(path).lower().replace("\\", "/")
    name = path.name.lower()
    if name in allowed or normalized in allowed:
        return None
    for pattern in SENSITIVE_NAMES:
        if fnmatch.fnmatch(name, pattern):
            return pattern
    return next((part for part in SENSITIVE_PARTS if part in f"/{normalized.lstrip('/')}"), None)


def command_match(command: str, allowed: set[str]) -> str | None:
    match = COMMAND_PATTERN.search(command)
    if not match:
        return None
    value = match.group(1).lower().replace("\\", "/")
    return None if value in allowed or Path(value).name in allowed else match.group(1)


def deny(event: str, reason: str) -> None:
    if event == "PermissionRequest":
        emit({"hookSpecificOutput": {
            "hookEventName": event,
            "decision": {"behavior": "deny", "message": reason},
        }})
        return
    emit({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }})


def main() -> None:
    payload = read_payload()
    root = project_root(payload.get("cwd"))
    config = load_config(root)
    if (config.get("privacyBlock") or {}).get("enabled", True) is False:
        return

    allowed = allow_list(config)
    for path in touched_paths(payload):
        matched = path_match(path, allowed)
        if matched:
            deny(payload.get("hook_event_name", "PreToolUse"), f"Sensitive path blocked ({matched}): {path.name}")
            return

    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
    if payload.get("tool_name") == "Bash" and isinstance(command, str):
        matched = command_match(command, allowed)
        if matched:
            deny(payload.get("hook_event_name", "PreToolUse"), f"Command references sensitive data: {matched}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(0)
