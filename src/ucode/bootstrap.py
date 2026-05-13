"""Best-effort runtime/bootstrap installer for ucode dependencies."""

from __future__ import annotations

from ucode.agents import TOOL_SPECS, ensure_bootstrap_dependencies
from ucode.ui import print_err


def main() -> int:
    try:
        for tool in TOOL_SPECS:
            ensure_bootstrap_dependencies(tool)
    except RuntimeError as exc:
        print_err(f"ucode bootstrap failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
