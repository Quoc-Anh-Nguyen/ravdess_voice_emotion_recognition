"""Privacy guard hook for Antigravity: blocks access to sensitive files and credentials."""

from __future__ import annotations

import fnmatch
import json
import re
import sys
from pathlib import Path

SENSITIVE_NAMES = (
    ".env", ".env.*", "*.key", "*.pem", "*.pfx", "*.p12", "*.keystore",
    "*.jks", "*credential*", "*secret*", "*password*", "*passwd*", "*.cert", "*.crt",
    "id_rsa", "id_ecdsa", "id_ed25519", "id_dsa",
)
SENSITIVE_PARTS = ("/.ssh/", "/secrets/", "/credentials/", "/.gnupg/")

COMMAND_PATTERN = re.compile(
    r"(?:^|[\\/\s'\"])(\.env(?:\.[\w.-]+)?|[^\s'\"]*\.(?:key|pem|pfx|p12|cert|crt)|"
    r"id_(?:rsa|ecdsa|ed25519|dsa)|[^\s'\"]*(?:credential|secret|password|passwd)[^\s'\"]*)",
    re.IGNORECASE,
)


def is_sensitive_path(path_str: str) -> str | None:
    path = Path(path_str)
    name = path.name.lower()
    normalized = str(path).lower().replace("\\", "/")

    for pattern in SENSITIVE_NAMES:
        if fnmatch.fnmatch(name, pattern):
            return name

    for part in SENSITIVE_PARTS:
        if part in f"/{normalized.lstrip('/')}":
            return part

    return None


def is_sensitive_command(command: str) -> str | None:
    match = COMMAND_PATTERN.search(command)
    if match:
        return match.group(1)
    return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    tool_call = payload.get("toolCall") or {}
    args = tool_call.get("args") or {}

    # Check file path arguments
    for key in ("AbsolutePath", "TargetFile", "file_path", "path"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            matched = is_sensitive_path(val.strip())
            if matched:
                print(json.dumps({
                    "decision": "deny",
                    "reason": f"Access to sensitive target '{matched}' is blocked by Privacy Guard."
                }, ensure_ascii=False))
                return

    # Check command execution
    for key in ("CommandLine", "command"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            matched = is_sensitive_command(val.strip())
            if matched:
                print(json.dumps({
                    "decision": "deny",
                    "reason": f"Command targeting sensitive pattern '{matched}' is blocked by Privacy Guard."
                }, ensure_ascii=False))
                return

    # Allow by default
    print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()

