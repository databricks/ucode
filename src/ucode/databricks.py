"""Databricks workspace integration: CLI auth, token retrieval, model
discovery, AI Gateway v2 enforcement, SQL warehouse discovery, URL builders."""

from __future__ import annotations

import functools
import json
import logging
import logging.handlers
import os
import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Literal, cast, overload
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from ucode.config_io import APP_DIR
from ucode.ui import (
    err_console,
    normalize_workspace_url,
    print_kv,
    print_note,
    print_section,
    print_success,
    print_warning,
    spinner,
)

UNIX_DATABRICKS_INSTALL_URL = (
    "https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh"
)
WINDOWS_DATABRICKS_INSTALL_URL = (
    "https://raw.githubusercontent.com/databricks/setup-cli/main/install.ps1"
)
AI_GATEWAY_V2_DOCS_URL = "https://docs.databricks.com/aws/en/ai-gateway/overview-beta"
MIN_DATABRICKS_CLI_VERSION = (0, 298, 0)
TOKEN_REFRESH_INTERVAL_SECONDS = 1800


def _debug_enabled() -> bool:
    return os.environ.get("UCODE_DEBUG") == "1"


_DEBUG_LOGGER: logging.Logger | None = None


def _get_debug_logger() -> logging.Logger | None:
    """Lazily configure a rotating file logger when UCODE_DEBUG=1.

    Returns the logger on first call (and caches it), or None if debug is
    disabled or the log file could not be opened. A one-time breadcrumb is
    printed to stderr so the user knows where to tail."""
    global _DEBUG_LOGGER
    if _DEBUG_LOGGER is not None or not _debug_enabled():
        return _DEBUG_LOGGER

    log_path = APP_DIR / "debug.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        )
    except OSError:
        return None

    logger = logging.getLogger("ucode.debug")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False
    _DEBUG_LOGGER = logger
    err_console.print(f"[dim]\\[ucode debug] logging to {log_path}[/dim]")
    return _DEBUG_LOGGER


def _debug(label: str, detail: str) -> None:
    """When UCODE_DEBUG=1, append a timestamped entry to ~/.ucode/debug.log."""
    logger = _get_debug_logger()
    if logger is not None:
        logger.debug("%s: %s", label, detail)


_SECRET_KEY_PATTERN = re.compile(r"(token|secret|password|bearer|api_key|apikey)", re.IGNORECASE)


def _format_subprocess_result(
    result: subprocess.CompletedProcess[str],
) -> str:
    """Format a CompletedProcess for the debug log without leaking tokens.

    On success, stdout is suppressed (it often contains the access token).
    On failure, stdout/stderr are included truncated."""
    stderr = (result.stderr or "").strip()[:500]
    if result.returncode == 0:
        return f"rc=0 stderr={stderr!r}"
    stdout = (result.stdout or "").strip()[:500]
    return f"rc={result.returncode} stdout={stdout!r} stderr={stderr!r}"


def _scrub_databrickscfg(text: str) -> str:
    """Redact value of any INI key that looks secret-bearing."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if "=" in stripped and not stripped.startswith(("#", ";")):
            key = stripped.split("=", 1)[0].strip()
            if _SECRET_KEY_PATTERN.search(key):
                indent = line[: len(line) - len(stripped)]
                out.append(f"{indent}{key} = <redacted>")
                continue
        out.append(line)
    return "\n".join(out)


def _scrub_json(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: (
                "<redacted>"
                if isinstance(k, str) and _SECRET_KEY_PATTERN.search(k)
                else _scrub_json(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_scrub_json(v) for v in value]
    return value


@functools.cache
def _log_auth_diagnostics() -> None:
    """Dump CLI version, profiles, and ~/.databrickscfg (scrubbed) to the debug log.

    No-op unless UCODE_DEBUG=1; cached so it runs at most once per process."""
    if not _debug_enabled():
        return

    try:
        version_result = subprocess.run(
            ["databricks", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = (version_result.stdout or version_result.stderr or "").strip()
        _debug("databricks --version", version[:200])
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("databricks --version", f"exception: {type(exc).__name__}: {exc}")

    try:
        profiles_result = subprocess.run(
            ["databricks", "auth", "profiles", "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        _debug(
            "databricks auth profiles",
            f"rc={profiles_result.returncode} "
            f"stderr={(profiles_result.stderr or '').strip()[:300]!r}",
        )
        if profiles_result.returncode == 0 and profiles_result.stdout:
            try:
                payload = json.loads(profiles_result.stdout)
                _debug("profiles json", json.dumps(_scrub_json(payload))[:2000])
            except json.JSONDecodeError as exc:
                _debug("profiles json", f"decode error: {exc}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("databricks auth profiles", f"exception: {type(exc).__name__}: {exc}")

    cfg_path = Path(os.environ.get("DATABRICKS_CONFIG_FILE") or "~/.databrickscfg").expanduser()
    try:
        if cfg_path.is_file():
            raw = cfg_path.read_text(encoding="utf-8", errors="replace")
            _debug(f"databrickscfg ({cfg_path})", _scrub_databrickscfg(raw)[:4000])
        else:
            _debug(f"databrickscfg ({cfg_path})", "not present")
    except OSError as exc:
        _debug(f"databrickscfg ({cfg_path})", f"read error: {exc}")


def _http_get_json(
    url: str, token: str, *, timeout: int = 10
) -> tuple[dict | list | None, str | None]:
    """GET a JSON endpoint. Returns (payload, None) on success, (None, reason) on failure.

    Honors UCODE_DEBUG=1 to append status + truncated body to ~/.ucode/debug.log.
    """
    request = urllib_request.Request(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        _debug(f"GET {url}", f"HTTP 200, {len(body)} bytes")
        if _debug_enabled():
            _debug("body", body[:4000])
        try:
            return json.loads(body), None
        except json.JSONDecodeError as exc:
            return None, f"response was not valid JSON ({exc.msg})"
    except urllib_error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        except Exception:
            body = ""
        _debug(f"GET {url}", f"HTTP {exc.code} {exc.reason}")
        if _debug_enabled() and body:
            _debug("body", body[:4000])
        return None, f"HTTP {exc.code} {exc.reason}"
    except urllib_error.URLError as exc:
        _debug(f"GET {url}", f"URLError: {exc.reason}")
        return None, f"network error: {exc.reason}"


@overload
def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: Literal[True],
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]: ...


@overload
def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: Literal[False] = False,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[bytes]: ...


def run(
    args: list[str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = False,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        check=check,
        capture_output=capture_output,
        text=text,
        env=env,
        timeout=timeout,
    )


def build_databricks_cli_env(workspace: str) -> dict[str, str]:
    env = os.environ.copy()
    env["DATABRICKS_HOST"] = workspace
    return env


def workspace_hostname(workspace: str) -> str:
    parsed = urlparse(normalize_workspace_url(workspace))
    if not parsed.hostname:
        raise RuntimeError(f"Unable to derive hostname from workspace URL: {workspace}")
    return parsed.hostname


def _parse_databricks_cli_version(output: str) -> tuple[int, int, int] | None:
    # Example output: "Databricks CLI v0.299.2"
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", output)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _run_databricks_cli_installer(brew_subcommand: str = "install") -> None:
    system = platform.system()
    try:
        if system == "Windows":
            run(
                ["powershell", "-Command", f"irm {WINDOWS_DATABRICKS_INSTALL_URL} | iex"],
                timeout=240,
            )
        elif system == "Darwin" and shutil.which("brew"):
            run(["brew", brew_subcommand, "databricks"], timeout=240)
        elif shutil.which("curl"):
            run(["sh", "-c", f"curl -fsSL {UNIX_DATABRICKS_INSTALL_URL} | sudo sh"], timeout=240)
        elif shutil.which("wget"):
            run(["sh", "-c", f"wget -qO- {UNIX_DATABRICKS_INSTALL_URL} | sudo sh"], timeout=240)
        else:
            raise RuntimeError("Neither curl nor wget is available.")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
        raise RuntimeError("Failed to install/upgrade Databricks CLI automatically.") from exc


def ensure_databricks_cli_version() -> None:
    try:
        result = run(
            ["databricks", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Failed to read Databricks CLI version.") from exc

    raw = result.stdout or result.stderr or ""
    output = (raw if isinstance(raw, str) else raw.decode(errors="replace")).strip()
    version = _parse_databricks_cli_version(output)
    if version is None:
        raise RuntimeError(
            f"Could not parse Databricks CLI version from `databricks --version` output: {output!r}"
        )
    if version < MIN_DATABRICKS_CLI_VERSION:
        current = ".".join(str(n) for n in version)
        required = ".".join(str(n) for n in MIN_DATABRICKS_CLI_VERSION)
        print_warning(
            f"Databricks CLI v{current} is too old (need v{required} or newer). Upgrading..."
        )
        _run_databricks_cli_installer(brew_subcommand="upgrade")
        ensure_databricks_cli_version()


def install_databricks_cli() -> None:
    if shutil.which("databricks"):
        ensure_databricks_cli_version()
        return

    print_section("Bootstrap")
    print_warning("`databricks` was not found. Installing Databricks CLI...")
    _run_databricks_cli_installer(brew_subcommand="install")

    if not shutil.which("databricks"):
        raise RuntimeError(
            "Databricks CLI install completed, but `databricks` is still not on PATH."
        )
    ensure_databricks_cli_version()


def _profile_args(profile: str | None) -> list[str]:
    """Return ``["--profile", profile]`` when set, otherwise an empty list.

    Centralizing this keeps every `databricks` CLI invocation in this module
    consistent when a workspace's `~/.databrickscfg` has more than one profile
    pointing at the same host."""
    return ["--profile", profile] if profile else []


def has_valid_databricks_auth(workspace: str, profile: str | None = None) -> bool:
    # Honor the CI short-circuit (see ``get_databricks_token``): if a
    # pre-fetched bearer is available, treat auth as valid and skip the
    # `databricks auth token` shell-out (which only knows user-OAuth).
    if os.environ.get("DATABRICKS_BEARER", "").strip():
        return True
    _log_auth_diagnostics()
    try:
        env = build_databricks_cli_env(workspace)
        result = run(
            [
                "databricks",
                "auth",
                "token",
                "--host",
                workspace,
                *_profile_args(profile),
                "--output",
                "json",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        _debug(
            "has_valid_databricks_auth",
            _format_subprocess_result(result),
        )
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout or "{}")
        return bool(data.get("access_token"))
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired) as exc:
        _debug("has_valid_databricks_auth", f"exception: {type(exc).__name__}: {exc}")
        return False


def get_databricks_profiles() -> list[tuple[str, str]]:
    """Return [(host_url, profile_name), ...] from Databricks CLI profiles.

    Returns ``[]`` on any failure (CLI missing, timeout, non-zero exit, JSON
    decode error). When ``UCODE_DEBUG=1`` each dropout path logs *why* the
    result was empty so a silently-disappearing workspace picker is
    diagnosable from ``~/.ucode/debug.log``.
    """
    try:
        result = run(
            ["databricks", "auth", "profiles", "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _debug("get_databricks_profiles", f"subprocess error: {type(exc).__name__}: {exc}")
        return []
    if result.returncode != 0:
        _debug("get_databricks_profiles", _format_subprocess_result(result))
        return []
    try:
        profiles = json.loads(result.stdout or "{}").get("profiles") or []
    except json.JSONDecodeError as exc:
        _debug("get_databricks_profiles", f"json decode error: {exc.msg}")
        return []

    # dict dedupes by host (first non-PAT profile wins).
    out: dict[str, str] = {}
    pat = 0
    for p in profiles:
        host = (p.get("host") or "").rstrip("/")
        name = p.get("name")
        if not host or not name:
            continue
        if p.get("auth_type") == "pat":
            pat += 1
            continue
        out.setdefault(host, name)

    _debug(
        "get_databricks_profiles",
        f"returned={len(out)} total={len(profiles)} pat={pat}",
    )
    return list(out.items())


def find_profile_name_for_host(workspace: str) -> str | None:
    """Find the Databricks CLI profile name matching a workspace URL."""
    normalized = workspace.rstrip("/")
    for host, name in get_databricks_profiles():
        if host == normalized:
            return name
    return None


def run_databricks_login(workspace: str, profile: str | None = None) -> None:
    """Run databricks auth login unconditionally.

    When ``profile`` is provided, it is passed via ``--profile``. Otherwise we
    fall back to looking up an existing profile by host so a stored session is
    refreshed in place rather than overwriting another profile's tokens."""
    print_section("Databricks Login")
    print_kv("Workspace", workspace)
    print_note("A browser may open for `databricks auth login`.")
    try:
        profile_name = profile or find_profile_name_for_host(workspace)
        cmd = [
            "databricks",
            "auth",
            "login",
            "--host",
            workspace,
            *_profile_args(profile_name),
        ]
        run(cmd, env=build_databricks_cli_env(workspace), timeout=300)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("`databricks auth login` failed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`databricks auth login` timed out.") from exc
    print_success("Databricks authentication complete")


def ensure_databricks_auth(workspace: str, profile: str | None = None) -> None:
    """Check auth and login only if needed (used by launch path)."""
    with spinner("Checking Databricks auth..."):
        auth_is_valid = has_valid_databricks_auth(workspace, profile)
    if auth_is_valid:
        print_success(f"Databricks auth already available for {workspace}")
        return
    run_databricks_login(workspace, profile)


def get_databricks_token(
    workspace: str,
    profile: str | None = None,
    *,
    force_refresh: bool = False,
) -> str:
    # ``DATABRICKS_BEARER`` is the CI escape hatch: when set, skip the
    # `databricks auth token` subprocess entirely and return the pre-fetched
    # bearer directly. Used by the e2e job, where the protected runner has
    # no `databricks auth login` cache and `databricks auth token` only knows
    # how to read user-OAuth caches (not M2M client_credentials). Mirrors the
    # same short-circuit baked into ``build_auth_shell_command``.
    bearer = os.environ.get("DATABRICKS_BEARER", "").strip()
    if bearer:
        _debug("get_databricks_token", "using DATABRICKS_BEARER env var")
        return bearer

    _log_auth_diagnostics()
    env = build_databricks_cli_env(workspace)
    cmd = [
        "databricks",
        "auth",
        "token",
        "--host",
        workspace,
        *_profile_args(profile),
        "--output",
        "json",
    ]
    if force_refresh:
        cmd.append("--force-refresh")

    _debug(
        "get_databricks_token.env",
        "set="
        + ",".join(sorted(k for k in env if k.startswith("DATABRICKS_") or k in {"BUNDLE_PROFILE"}))
        + f" profile={profile or '<none>'}",
    )

    def _fetch() -> str:
        try:
            result = run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
            _debug("auth token", _format_subprocess_result(result))
            if result.returncode == 0:
                return json.loads(result.stdout or "{}").get("access_token", "")
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            _debug("auth token", f"exception: {type(exc).__name__}: {exc}")
        return ""

    token = _fetch()
    if not token:
        # Session may have expired — attempt non-interactive re-auth and retry once.
        _debug("auth token", "empty on first fetch; attempting auth login --no-browser")
        try:
            reauth = run(
                [
                    "databricks",
                    "auth",
                    "login",
                    "--host",
                    workspace,
                    *_profile_args(profile),
                    "--no-browser",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            _debug("auth login", _format_subprocess_result(reauth))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            _debug("auth login", f"exception: {type(exc).__name__}: {exc}")
        token = _fetch()

    if not token:
        profile_name = profile or find_profile_name_for_host(workspace)
        stale_profile_hint = ""
        if profile_name:
            stale_profile_hint = (
                " The saved Databricks CLI profile may be stale or invalid. Try:\n"
                f"  databricks auth logout --profile {profile_name}\n"
                f"  databricks auth login --host {workspace} --profile {profile_name}"
            )
        raise RuntimeError(
            f"Databricks CLI returned no access token for {workspace}. "
            "Run `databricks auth login` to re-authenticate."
            f"{stale_profile_hint}"
        )
    return token


def _extract_connection_page(payload: object) -> tuple[list[dict], str | None]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)], None
    if not isinstance(payload, dict):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    payload_dict = cast(dict[str, object], payload)
    raw_connections = payload_dict.get("connections") or []
    if not isinstance(raw_connections, list):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    next_page_token = payload_dict.get("next_page_token")
    if next_page_token is not None and not isinstance(next_page_token, str):
        raise RuntimeError("Databricks connections listing returned invalid JSON.")

    return [item for item in raw_connections if isinstance(item, dict)], next_page_token


def list_databricks_connections(workspace: str) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    connections: list[dict] = []
    page_token: str | None = None
    seen_page_tokens: set[str] = set()

    try:
        while True:
            cmd = [
                "databricks",
                "connections",
                "list",
                "--max-results",
                "0",
                "--output",
                "json",
            ]
            if page_token:
                cmd.extend(["--page-token", page_token])

            result = run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            payload = json.loads(result.stdout or "{}")
            page_connections, page_token = _extract_connection_page(payload)
            connections.extend(page_connections)

            if not page_token:
                return connections
            if page_token in seen_page_tokens:
                raise RuntimeError("Databricks connections listing returned a repeated page token.")
            seen_page_tokens.add(page_token)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to list Databricks connections via `databricks connections list`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks connections.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks connections listing returned invalid JSON.") from exc


def _extract_genie_spaces_page(payload: object) -> tuple[list[dict], str | None]:
    if not isinstance(payload, dict):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    payload_dict = cast(dict[str, object], payload)
    raw_spaces = payload_dict.get("spaces") or []
    if not isinstance(raw_spaces, list):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    next_page_token = payload_dict.get("next_page_token")
    if next_page_token is not None and not isinstance(next_page_token, str):
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.")

    return [item for item in raw_spaces if isinstance(item, dict)], next_page_token


def list_genie_spaces(workspace: str) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    spaces: list[dict] = []
    page_token: str | None = None
    seen_page_tokens: set[str] = set()

    try:
        while True:
            cmd = [
                "databricks",
                "genie",
                "list-spaces",
                "--page-size",
                "100",
                "--output",
                "json",
            ]
            if page_token:
                cmd.extend(["--page-token", page_token])

            result = run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            payload = json.loads(result.stdout or "{}")
            page_spaces, page_token = _extract_genie_spaces_page(payload)
            spaces.extend(page_spaces)

            if not page_token:
                return spaces
            if page_token in seen_page_tokens:
                raise RuntimeError(
                    "Databricks Genie spaces listing returned a repeated page token."
                )
            seen_page_tokens.add(page_token)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to list Databricks Genie spaces via `databricks genie list-spaces`."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks Genie spaces.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks Genie spaces listing returned invalid JSON.") from exc


def _extract_apps_payload(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        payload_dict = cast(dict[str, object], payload)
        raw_apps = payload_dict.get("apps") or []
        if isinstance(raw_apps, list):
            return [item for item in raw_apps if isinstance(item, dict)]
    raise RuntimeError("Databricks apps listing returned invalid JSON.")


def list_databricks_apps(workspace: str) -> list[dict]:
    env = build_databricks_cli_env(workspace)
    try:
        result = run(
            [
                "databricks",
                "apps",
                "list",
                "--limit",
                "1000",
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        return _extract_apps_payload(json.loads(result.stdout or "[]"))
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Failed to list Databricks apps via `databricks apps list`.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Timed out while listing Databricks apps.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks apps listing returned invalid JSON.") from exc


def build_auth_shell_command(workspace: str, profile: str | None = None) -> str:
    workspace_arg = shlex.quote(workspace.rstrip("/"))
    if profile:
        profile_arg = shlex.quote(profile)
        cli_command = (
            f"databricks auth token --host {workspace_arg} "
            f"--profile {profile_arg} --force-refresh --output json "
            "| jq -r '.access_token'"
        )
    else:
        cli_command = (
            "env -u DATABRICKS_CONFIG_PROFILE "
            f"databricks auth token --host {workspace_arg} --force-refresh --output json "
            "| jq -r '.access_token'"
        )
    return (
        'if [ -n "${DATABRICKS_BEARER:-}" ]; then '
        'printf "%s\\n" "$DATABRICKS_BEARER"; '
        f"else {cli_command}; fi"
    )


def discover_claude_models(workspace: str, token: str) -> tuple[dict[str, str], str | None]:
    """Discover Claude families on this workspace's AI Gateway.

    Returns (models_by_family, reason). reason is None on success; otherwise it
    describes why the dict is empty (HTTP error, network error, or no models
    matching the expected naming convention).
    """
    hostname = workspace_hostname(workspace)
    payload, reason = _http_get_json(f"https://{hostname}/ai-gateway/anthropic/v1/models", token)
    if payload is None:
        return {}, reason

    data = cast(dict, payload) if isinstance(payload, dict) else {}
    raw_ids = [
        m["id"]
        for m in data.get("data", [])
        if isinstance(m.get("id"), str) and not m["id"].endswith("-anthropic")
    ]

    result: dict[str, str] = {}
    for family, key in [("opus", "opus"), ("sonnet", "sonnet"), ("haiku", "haiku")]:
        candidates = sorted(
            [m for m in raw_ids if f"databricks-claude-{family}-" in m],
            reverse=True,
        )
        if candidates:
            result[key] = candidates[0]
    if result:
        return result, None
    if not raw_ids:
        return {}, "AI Gateway returned no Claude model ids"
    sample = ", ".join(raw_ids[:5])
    return {}, (
        "AI Gateway returned model ids but none matched "
        f"`databricks-claude-{{opus,sonnet,haiku}}-*` (got: {sample})"
    )


def fetch_ai_gateway_claude_models(workspace: str, token: str) -> dict[str, str]:
    """Backwards-compatible wrapper that discards the diagnostic reason."""
    models, _ = discover_claude_models(workspace, token)
    return models


def discover_endpoints_with_api_type(
    workspace: str, token: str, api_type: str
) -> tuple[list[str], str | None]:
    """List endpoint names whose served_entities expose api_type with v2 support.

    Returns (endpoints, reason). reason is None on success; otherwise it
    describes why the list is empty.
    """
    hostname = workspace_hostname(workspace)
    payload, reason = _http_get_json(
        f"https://{hostname}/api/2.0/serving-endpoints:foundation-models", token
    )
    if payload is None:
        return [], reason

    data = cast(dict, payload) if isinstance(payload, dict) else {}
    endpoints = data.get("endpoints", [])
    out: list[str] = []
    saw_endpoint_without_v2 = False
    for ep in endpoints:
        name = ep.get("name", "")
        entities = ep.get("config", {}).get("served_entities", [])
        api_types: set[str] = set()
        any_v2 = False
        for se in entities:
            fm = se.get("foundation_model", {})
            if fm.get("ai_gateway_v2_supported") is True:
                any_v2 = True
                api_types.update(fm.get("api_types", []))
        if not any_v2 and entities:
            saw_endpoint_without_v2 = True
        if api_type in api_types:
            out.append(name)
    if out:
        return sorted(out), None
    if not endpoints:
        return [], "foundation-models listing returned no endpoints"
    if saw_endpoint_without_v2:
        return [], (
            f"no endpoint exposes api_type `{api_type}` with "
            "`ai_gateway_v2_supported=true` (workspace has v1-only endpoints)"
        )
    return [], f"no endpoint exposes api_type `{api_type}`"


def _fetch_endpoints_with_api_type(workspace: str, token: str, api_type: str) -> list[str]:
    """Backwards-compatible wrapper that discards the diagnostic reason."""
    endpoints, _ = discover_endpoints_with_api_type(workspace, token, api_type)
    return endpoints


def discover_gemini_models(workspace: str, token: str) -> tuple[list[str], str | None]:
    return discover_endpoints_with_api_type(workspace, token, "gemini/v1/generateContent")


def discover_codex_models(workspace: str, token: str) -> tuple[list[str], str | None]:
    return discover_endpoints_with_api_type(workspace, token, "openai/v1/responses")


def fetch_gemini_models(workspace: str, token: str) -> list[str]:
    models, _ = discover_gemini_models(workspace, token)
    return models


def fetch_codex_models(workspace: str, token: str) -> list[str]:
    models, _ = discover_codex_models(workspace, token)
    return models


def ensure_ai_gateway_v2(workspace: str, token: str) -> None:
    """Probe AI Gateway v2 and raise if unavailable.

    Uses the dedicated v2 listing endpoint `GET /api/ai-gateway/v2/endpoints`:
    a 200 response (even with an empty list) means v2 is wired up on this
    workspace — a "no endpoints provisioned" case will surface naturally in
    downstream discovery. 404 / 401 / 403 / network failures all raise a
    clear error with the docs link instead of silently progressing.
    """
    hostname = workspace_hostname(workspace)
    url = f"https://{hostname}/api/ai-gateway/v2/endpoints?page_size=1"
    payload, reason = _http_get_json(url, token)
    if payload is not None:
        return
    raise RuntimeError(
        "Databricks AI Gateway V2 is required but not available on this workspace "
        f"({reason}). See {AI_GATEWAY_V2_DOCS_URL}"
    )


def discover_sql_warehouse_http_path(
    workspace: str,
    token: str,
    *,
    quiet: bool = False,
) -> str:
    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/api/2.0/sql/warehouses",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )

    try:
        with urllib_request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        detail = body.strip() or f"HTTP {exc.code}"
        raise RuntimeError(f"Failed to list SQL warehouses: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Could not reach workspace hostname {hostname}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Databricks warehouse discovery returned invalid JSON.") from exc

    warehouses = payload.get("warehouses")
    if not isinstance(warehouses, list) or not warehouses:
        raise RuntimeError(
            "No SQL warehouses found in this workspace. Create one or pass `--http-path`."
        )

    running = [w for w in warehouses if isinstance(w, dict) and w.get("state") == "RUNNING"]
    chosen = (
        running[0]
        if running
        else next(
            (w for w in warehouses if isinstance(w, dict) and w.get("id")),
            None,
        )
    )
    if not chosen:
        raise RuntimeError("No usable SQL warehouse was returned by Databricks.")

    warehouse_id = chosen.get("id")
    if not isinstance(warehouse_id, str) or not warehouse_id.strip():
        raise RuntimeError("Databricks returned a warehouse without an ID.")

    warehouse_name = chosen.get("name")
    warehouse_state = chosen.get("state", "UNKNOWN")
    label_value = (
        warehouse_name if isinstance(warehouse_name, str) and warehouse_name else warehouse_id
    )
    if not quiet:
        print_note(f"Using SQL warehouse `{label_value}` ({warehouse_state}).")
    return f"/sql/1.0/warehouses/{warehouse_id}"


def run_usage_query(
    workspace: str,
    http_path: str,
    token: str,
    query: str,
) -> tuple[list[str], list[tuple]]:
    try:
        logging.getLogger("databricks.sql").setLevel(logging.ERROR)
        from databricks import sql
    except ImportError as exc:
        raise RuntimeError(
            "`databricks-sql-connector` is not installed. "
            "Install it with `pip install databricks-sql-connector`."
        ) from exc

    try:
        with sql.connect(
            server_hostname=workspace_hostname(workspace),
            http_path=http_path,
            access_token=token,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [desc[0] for desc in (cursor.description or [])]
                rows = cast(list[tuple], cursor.fetchall())
    except Exception as exc:
        raise RuntimeError(f"Usage query failed: {exc}") from exc

    return columns, rows


# ---------------------------------------------------------------------------
# URL builders (AI Gateway v2 only — no fallback to /serving-endpoints)
# ---------------------------------------------------------------------------


def build_tool_base_url(tool: str, workspace: str) -> str:
    if tool == "codex":
        return f"{workspace}/ai-gateway/codex/v1"
    if tool == "claude":
        return f"{workspace}/ai-gateway/anthropic"
    if tool == "gemini":
        return f"{workspace}/ai-gateway/gemini"
    if tool == "opencode":
        raise RuntimeError(
            "OpenCode has multiple base URLs — use build_opencode_base_urls() instead."
        )
    if tool == "copilot":
        raise RuntimeError(
            "Copilot has multiple base URLs — use build_copilot_base_urls() instead."
        )
    if tool == "pi":
        raise RuntimeError("Pi has multiple base URLs — use build_pi_base_urls() instead.")
    raise RuntimeError(f"Unsupported tool '{tool}'.")


def build_opencode_base_urls(workspace: str) -> dict[str, str]:
    return {
        "anthropic": build_tool_base_url("claude", workspace) + "/v1",
        "gemini": build_tool_base_url("gemini", workspace) + "/v1beta",
    }


def build_pi_base_urls(workspace: str) -> dict[str, str]:
    # Pi speaks each model family's native API dialect to its dedicated gateway
    # path (verified end-to-end). Each `api` type appends its own path suffix:
    #
    # - anthropic-messages       appends `/v1/messages`
    # - openai-responses         appends `/responses`
    # - google-generative-ai     appends `/v1beta/models/{id}:streamGenerateContent`
    # - openai-completions       appends `/chat/completions`
    #
    # So the baseUrls below stop just before the suffix Pi will tack on.
    # Compat flags applied per-provider in agents/pi.py; required for `oss`
    # only (MLflow rejects `store` and `tools[].function.strict`).
    return {
        "claude": build_tool_base_url("claude", workspace),
        "openai": build_tool_base_url("codex", workspace),
        "gemini": build_tool_base_url("gemini", workspace) + "/v1beta",
    }


def build_copilot_base_url(workspace: str) -> str:
    # Copilot CLI's `openai` provider appends `/chat/completions` to the
    # configured base URL. The Databricks MLflow chat-completions gateway is
    # OpenAI-compatible and serves Claude, codex (gpt-5), and gemini models
    # behind one URL.
    return f"{workspace}/ai-gateway/mlflow/v1"


def build_shared_base_urls(workspace: str) -> dict[str, str | dict[str, str]]:
    urls: dict[str, str | dict[str, str]] = {
        "codex": build_tool_base_url("codex", workspace),
        "claude": build_tool_base_url("claude", workspace),
        "gemini": build_tool_base_url("gemini", workspace),
        "opencode": build_opencode_base_urls(workspace),
        "copilot": build_copilot_base_url(workspace),
        "pi": build_pi_base_urls(workspace),
    }
    return urls
