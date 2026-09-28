"""Run bounded project checks after Codex edits relevant source files."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from common import additional_context, load_config, project_root, read_payload, touched_paths


SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".py"}


def run(command: list[str], cwd: Path, timeout: int) -> tuple[int, str]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, shell=False, check=False)
        return result.returncode, (result.stdout + result.stderr).strip()
    except FileNotFoundError:
        return 0, ""
    except subprocess.TimeoutExpired:
        return 124, f"Check timed out after {timeout}s."


def package_scripts(root: Path) -> dict:
    try:
        value = json.loads((root / "package.json").read_text(encoding="utf-8"))
        scripts = value.get("scripts") or {}
        return scripts if isinstance(scripts, dict) else {}
    except Exception:
        return {}


def main() -> None:
    payload = read_payload()
    paths = touched_paths(payload)
    if not paths or not any(path.suffix.lower() in SOURCE_SUFFIXES for path in paths):
        return

    root = project_root(payload.get("cwd"))
    scripts = package_scripts(root)
    failures: list[str] = []

    if "typecheck" in scripts and any(path.suffix.lower() in {".ts", ".tsx", ".js", ".jsx"} for path in paths):
        code, output = run(["npm", "run", "typecheck", "--", "--pretty", "false"], root, 40)
        if code != 0:
            failures.append(f"Typecheck failed:\n{output[-4000:]}")

    for path in paths:
        if path.suffix.lower() == ".py" and path.exists():
            code, output = run([sys.executable, "-m", "py_compile", str(path)], root, 15)
            if code != 0:
                failures.append(f"Python syntax check failed for {path.name}:\n{output[-2000:]}")

    config = load_config(root)
    if config.get("testRunner", False) and "test" in scripts:
        code, output = run(["npm", "test", "--", "--run"], root, 40)
        if code != 0:
            failures.append(f"Tests failed:\n{output[-4000:]}")

    if failures:
        additional_context("PostToolUse", "\n\n".join(failures) + "\nFix these issues before completion.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(0)
