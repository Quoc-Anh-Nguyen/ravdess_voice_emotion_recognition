"""Post-tool quality check hook for Antigravity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    tool_call = payload.get("toolCall") or {}
    args = tool_call.get("args") or {}

    target_file = args.get("TargetFile") or args.get("file_path") or args.get("path")
    if target_file and isinstance(target_file, str):
        path = Path(target_file)
        if path.suffix.lower() == ".py" and path.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "py_compile", str(path)],
                    capture_output=True,
                    timeout=15,
                    check=False
                )
            except Exception:
                pass

    # PostToolUse expects empty JSON on stdout
    print(json.dumps({}))


if __name__ == "__main__":
    main()

