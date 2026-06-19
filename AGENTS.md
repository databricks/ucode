# Agent Instructions

## Project

`voxcode` is Van Oord's thin OpenCode launcher through the Databricks AI Gateway.
Only OpenCode is supported. Models are restricted to a platform-team-maintained allowlist.

The package code lives in `src/voxcode/`.
Tests live in `tests/`.

## Commands

- Run the full test suite with `uv run pytest`.
- Run focused tests with `uv run pytest tests/<file>.py`.
- Run e2e tests with `UCODE_TEST_WORKSPACE=<db_workspace_url> uv run pytest tests/test_e2e.py -v`.
- Run lint with `uv run ruff check .`.
- Run the CLI from the current checkout with `uv run voxcode ...`.
- Reinstall the local checkout as the `voxcode` tool with `uv tool install --reinstall .`.

## Development

- Use Python 3.12+.
- Keep changes scoped to the requested behavior.
- Follow the existing module boundaries: CLI orchestration in `cli.py`, OpenCode agent in `agents/opencode.py`, shared dispatch in `agents/__init__.py`, Databricks calls in `databricks.py`, presentation in `ui.py`, model allowlist in `allowed_models.py`.
- Prefer existing helpers for config file writes, state persistence, UI messages, and Databricks authentication.
- Add or update focused tests for behavior changes.
- Do not modify generated or lock files unless the dependency graph intentionally changes.

## Style

- Keep user-facing CLI errors actionable.
- Use warnings for recoverable setup problems and errors for launch/runtime blockers.
- Preserve existing Rich UI conventions, including `print_warning`, `print_err`, `print_success`, `print_section`, and `spinner`.
- Avoid broad refactors while fixing a narrow bug.
