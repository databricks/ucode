"""Databricks workspace integration: CLI auth, token retrieval, model
discovery, AI Gateway v2 enforcement, SQL warehouse discovery, URL builders."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
from typing import cast
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from ucode.ui import (
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


def has_valid_databricks_auth(workspace: str) -> bool:
    try:
        env = build_databricks_cli_env(workspace)
        result = run(
            ["databricks", "auth", "token", "--host", workspace, "--output", "json"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout or "{}")
        return bool(data.get("access_token"))
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired):
        return False


def get_databricks_profiles() -> list[tuple[str, str]]:
    """Return [(host_url, profile_name), ...] from Databricks CLI profiles."""
    try:
        result = run(
            ["databricks", "auth", "profiles", "--output", "json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout or "{}")
        profiles = data.get("profiles") or []
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        for p in profiles:
            host = p.get("host", "").rstrip("/")
            if host and host not in seen and p.get("auth_type") != "pat":
                seen.add(host)
                out.append((host, p["name"]))
        return out
    except (json.JSONDecodeError, OSError, subprocess.TimeoutExpired, KeyError):
        return []


def find_profile_name_for_host(workspace: str) -> str | None:
    """Find the Databricks CLI profile name matching a workspace URL."""
    normalized = workspace.rstrip("/")
    for host, name in get_databricks_profiles():
        if host == normalized:
            return name
    return None


def run_databricks_login(workspace: str) -> None:
    """Run databricks auth login unconditionally."""
    print_section("Databricks Login")
    print_kv("Workspace", workspace)
    print_note("A browser may open for `databricks auth login`.")
    try:
        cmd = ["databricks", "auth", "login", "--host", workspace]
        profile_name = find_profile_name_for_host(workspace)
        if profile_name:
            cmd += ["--profile", profile_name]
        run(cmd, env=build_databricks_cli_env(workspace), timeout=300)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("`databricks auth login` failed.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("`databricks auth login` timed out.") from exc
    print_success("Databricks authentication complete")


def ensure_databricks_auth(workspace: str) -> None:
    """Check auth and login only if needed (used by launch path)."""
    with spinner("Checking Databricks auth..."):
        auth_is_valid = has_valid_databricks_auth(workspace)
    if auth_is_valid:
        print_success(f"Databricks auth already available for {workspace}")
        return
    run_databricks_login(workspace)


def get_databricks_token(workspace: str) -> str:
    env = build_databricks_cli_env(workspace)

    def _fetch() -> str:
        try:
            result = run(
                ["databricks", "auth", "token", "--host", workspace, "--output", "json"],
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
            return json.loads(result.stdout or "{}").get("access_token", "")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return ""

    token = _fetch()
    if not token:
        # Session may have expired — attempt non-interactive re-auth and retry once.
        try:
            run(
                ["databricks", "auth", "login", "--host", workspace, "--no-browser"],
                capture_output=True,
                env=env,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        token = _fetch()

    if not token:
        raise RuntimeError(
            f"Databricks CLI returned no access token for {workspace}. "
            "Run `databricks auth login` to re-authenticate."
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


def build_auth_shell_command(workspace: str) -> str:
    return (
        f"databricks auth token --host {workspace} --force-refresh --output json "
        f"| jq -r '.access_token'"
    )


def fetch_ai_gateway_claude_models(workspace: str, token: str) -> dict[str, str]:
    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/ai-gateway/anthropic/v1/models",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib_request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError):
        return {}

    models = [
        m["id"]
        for m in data.get("data", [])
        if isinstance(m.get("id"), str) and not m["id"].endswith("-anthropic")
    ]

    result: dict[str, str] = {}
    for family, key in [("opus", "opus"), ("sonnet", "sonnet"), ("haiku", "haiku")]:
        candidates = sorted(
            [m for m in models if f"databricks-claude-{family}-" in m],
            reverse=True,
        )
        if candidates:
            result[key] = candidates[0]
    return result


def _fetch_endpoints_with_api_type(workspace: str, token: str, api_type: str) -> list[str]:
    """Generic helper: list endpoint names whose served_entities expose api_type."""
    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/api/2.0/serving-endpoints:foundation-models",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib_request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, json.JSONDecodeError):
        return []

    out: list[str] = []
    for ep in data.get("endpoints", []):
        name = ep.get("name", "")
        entities = ep.get("config", {}).get("served_entities", [])
        api_types: set[str] = set()
        for se in entities:
            fm = se.get("foundation_model", {})
            if fm.get("ai_gateway_v2_supported") is True:
                api_types.update(fm.get("api_types", []))
        if api_type in api_types:
            out.append(name)
    return sorted(out)


def fetch_gemini_models(workspace: str, token: str) -> list[str]:
    return _fetch_endpoints_with_api_type(workspace, token, "gemini/v1/generateContent")


def fetch_codex_models(workspace: str, token: str) -> list[str]:
    return _fetch_endpoints_with_api_type(workspace, token, "openai/v1/responses")


def ensure_ai_gateway_v2(workspace: str, token: str) -> None:
    """Probe AI Gateway v2 and raise if unavailable.

    Replaces the prior detect/fall-back pattern: v2 is now mandatory.
    """
    hostname = workspace_hostname(workspace)
    request = urllib_request.Request(
        f"https://{hostname}/ai-gateway/anthropic/v1/messages",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        urllib_request.urlopen(request, timeout=10)
        return
    except urllib_error.HTTPError as exc:
        if exc.code != 404:
            return
    except urllib_error.URLError:
        pass
    raise RuntimeError(
        "Databricks AI Gateway V2 is required but not available on this workspace. "
        f"See {AI_GATEWAY_V2_DOCS_URL}"
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
    raise RuntimeError(f"Unsupported tool '{tool}'.")


def build_opencode_base_urls(workspace: str) -> dict[str, str]:
    return {
        "anthropic": build_tool_base_url("claude", workspace) + "/v1",
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
    }
    return urls
