"""Drive installed programs; never import the application under test."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def clean_environment(home: Path) -> dict[str, str]:
    # An allowlist prevents a developer's agent keys, settings, plugins, Python
    # imports, and routing flags from silently changing the tested combination.
    keep = (
        "PATH",
        "SYSTEMROOT",
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
    )
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_DATA_HOME": str(home / ".local/share"),
            "CLAUDE_CONFIG_DIR": str(home / ".claude"),
            "CODEX_HOME": str(home / ".codex"),
            "DATABRICKS_CONFIG_FILE": str(home / ".databrickscfg"),
            "DISABLE_AUTOUPDATER": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "COLUMNS": "160",
            "ENABLE_SMART_ROUTING_V2": "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return env


def stop_process(proc: subprocess.Popen) -> None:
    """Reap the entire process group, including servers left by a failed agent."""
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The leader may have exited while a grandchild kept running.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    elif proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


class UserSession:
    def __init__(self, root: Path, project_root: Path, binary: Path, artifacts: Path):
        self.home = root / "home"
        self.cwd = project_root / "project with spaces"
        self.home.mkdir(parents=True)
        self.cwd.mkdir()
        self.binary = binary
        self.env = clean_environment(self.home)
        self.artifacts = artifacts
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.commands = 0

    def redact(self, text: str) -> str:
        for token in (os.environ.get("DATABRICKS_BEARER"), self.env.get("DATABRICKS_BEARER")):
            if token:
                text = text.replace(token, "<redacted>")
        return ANSI.sub("", text)

    def run(
        self,
        *args: str,
        timeout: int = 120,
        ok: bool = True,
        binary=None,
        input_text: str | None = None,
    ):
        command = [str(binary or self.binary), *args]
        proc = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name == "posix",
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_process(proc)
            stdout, stderr = proc.communicate(timeout=5)
        finally:
            stop_process(proc)
        result = subprocess.CompletedProcess(
            command, proc.returncode, self.redact(stdout), self.redact(stderr)
        )
        self.commands += 1
        self.record(
            f"command-{self.commands}.json",
            {
                "argv": command,
                "returncode": result.returncode,
                "timed_out": timed_out,
                "stdin": input_text,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
        detail = f"{command}\n{result.stdout}\n{result.stderr}"
        assert not timed_out, f"Command exceeded {timeout}s:\n{detail}"
        if ok:
            assert result.returncode == 0, detail
        return result

    def record(self, name: str, value: object) -> None:
        (self.artifacts / name).write_text(self.redact(json.dumps(value, indent=2)))

    def configure(self, agent: str, workspace: str, *, ok: bool = True):
        return self.run(
            "configure",
            "--agents",
            agent,
            "--workspaces",
            workspace,
            "--skip-validate",
            "--skip-upgrade",
            "--disable-databricks-ai-tools",
            ok=ok,
        )

    def state(self) -> dict:
        return json.loads((self.home / ".ucode/state.json").read_text())

    def model_for_explicit_case(self, agent: str) -> str:
        """Use a real discovered model only when testing an explicit model option."""
        model = os.environ.get(f"UG_INTEGRATION_{agent.upper()}_MODEL", "").strip()
        source = "runner override"
        if not model:
            state = self.state()
            workspace = state["workspaces"][state["current_workspace"]]
            models = workspace[f"{agent}_models"]
            values = models.values() if isinstance(models, dict) else models
            prefix = "system.ai.claude-" if agent == "claude" else "system.ai.gpt-"
            model = next((value for value in values if value.startswith(prefix)), "")
            source = "ug configure discovery"
        assert model, (
            f"ug configure found no system.ai model for {agent}; use --{agent}-model to reproduce a specific model."
        )
        self.record("model.json", {"model": model, "source": source})
        return model

    def assert_not_routed(self) -> None:
        # These are user-visible diagnostics produced only by routing wrappers.
        for name in ("codex-v2-interposer.log", "claude-v2-pty.log"):
            assert not (self.home / ".ucode" / name).exists(), f"Unexpected routing: {name}"

    def app_server_handshake(self, args: list[str], timeout: int = 120) -> dict:
        """Speak the real Codex stdio protocol and require an initialize response."""
        command = [str(self.binary), "codex", *args]
        messages: queue.Queue = queue.Queue()
        transcript: list[str] = []
        diagnostics: list[str] = []
        proc = subprocess.Popen(
            command,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=os.name == "posix",
        )

        def read_output():
            for line in proc.stdout:
                transcript.append(self.redact(line))
                try:
                    messages.put(json.loads(line))
                except ValueError:
                    if line.strip():
                        messages.put({"protocol_error": self.redact(line)})
            messages.put(None)

        def read_diagnostics():
            for line in proc.stderr:
                diagnostics.append(self.redact(line))

        reader = threading.Thread(target=read_output, daemon=True)
        stderr_reader = threading.Thread(target=read_diagnostics, daemon=True)
        reader.start()
        stderr_reader.start()
        try:
            proc.stdin.write(
                json.dumps(
                    {
                        "id": 1,
                        "method": "initialize",
                        "params": {"clientInfo": {"name": "ug-integration", "version": "1.0.0"}},
                    }
                )
                + "\n"
            )
            proc.stdin.flush()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    message = messages.get(timeout=max(0.01, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if message is None:
                    break
                assert isinstance(message, dict), message
                assert "protocol_error" not in message, (
                    "Non-JSON output on the app-server protocol stream: " + str(message)
                )
                if isinstance(message, dict) and message.get("id") == 1:
                    assert "error" not in message, message
                    assert isinstance(message.get("result"), dict), message
                    assert message["result"].get("userAgent"), message
                    proc.stdin.write('{"method":"initialized","params":{}}\n')
                    proc.stdin.flush()
                    return message
            raise AssertionError("No app-server initialize response:\n" + "".join(transcript))
        finally:
            stop_process(proc)
            reader.join(timeout=5)
            stderr_reader.join(timeout=5)
            proc.stdin.close()
            proc.stdout.close()
            proc.stderr.close()
            self.record(
                "app-server.json", {"argv": command, "stdout": transcript, "stderr": diagnostics}
            )
