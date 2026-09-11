"""A real PTY and terminal screen for driving installed interactive agents.

The screen implements terminal device replies; it never substitutes an agent,
gateway, application function, configuration file, or model response.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import time
import uuid

import pexpect
import pyte


class TerminalScreen(pyte.Screen):
    def __init__(self, columns, lines, send):
        super().__init__(columns, lines)
        self.send = send

    def write_process_input(self, data):
        # Real TUIs query cursor position/device attributes during startup.
        # pyte replies according to the terminal state it has actually rendered.
        self.send(data)


class AgentTerminal:
    def __init__(self, session, agent, command, name):
        self.session = session
        self.agent = agent
        self.name = name
        self.command = command
        env = {**session.env, "TERM": "xterm-256color"}
        self.child = pexpect.spawn(
            self.command[0],
            self.command[1:],
            cwd=str(session.cwd),
            env=env,
            encoding="utf-8",
            codec_errors="replace",
            dimensions=(40, 140),
            timeout=120,
        )
        self.screen = TerminalScreen(140, 40, self.child.send)
        self.stream = pyte.Stream(self.screen)
        self.output = []
        self.actions = []
        self.ended = False

    @property
    def visible(self):
        return "\n".join(line.rstrip() for line in self.screen.display)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        # pexpect creates a new session with a controlling terminal. Clean up
        # its process group even if the ug leader exited before its children.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.child.pid, signal.SIGTERM)
        self.child.close(force=True)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.child.pid, signal.SIGKILL)
        routing_log = (
            self.session.home
            / ".ucode"
            / ("claude-v2-pty.log" if self.agent == "claude" else "codex-v2-interposer.log")
        )
        self.session.record(
            f"{self.name}.json",
            {
                "argv": self.command,
                "terminal": {"rows": 40, "columns": 140, "term": "xterm-256color"},
                "actions": self.actions,
                "transcript": "".join(self.output),
                "screen": self.visible,
                "exitstatus": self.child.exitstatus,
                "signalstatus": self.child.signalstatus,
                "normal_exit_observed": self.ended and self.child.exitstatus == 0,
                "routing_log": routing_log.read_text() if routing_log.is_file() else None,
            },
        )

    def read(self):
        try:
            chunk = self.child.read_nonblocking(size=65536, timeout=0.2)
        except pexpect.TIMEOUT:
            return
        except pexpect.EOF:
            self.ended = True
            return
        self.output.append(chunk)
        self.stream.feed(chunk)

    def send(self, keys, reason):
        self.actions.append({"reason": reason, "keys": keys, "screen_before": self.visible})
        self.child.send(keys)

    def wait_for(self, predicate, description, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.read()
            assert not self.ended, f"TUI exited while waiting for {description}:\n{self.visible}"
            if predicate(self.visible):
                return
        raise AssertionError(f"TUI did not show {description} within {timeout}s:\n{self.visible}")

    def boot(self, timeout=120):
        """Handle only recognized visible onboarding; unknown screens fail."""
        handled = set()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.read()
            assert not self.ended, f"TUI exited before its prompt:\n{self.visible}"
            text = self.visible
            # These are interactions with ordinary UI choices, not pre-written
            # onboarding state. Only the test's disposable project is trusted.
            dialogs = [
                (
                    "theme",
                    "Choose the text style" in text and "Dark mode" in text,
                    "\r",
                ),
                (
                    "security-notes",
                    "Security notes" in text and "Enter to continue" in text,
                    "\r",
                ),
                (
                    "trust-folder",
                    self.session.cwd.name in text
                    and bool(re.search(r"1[.)]\s+Yes, I trust (?:this|the) folder", text)),
                    "1\r",
                ),
                (
                    "trust-directory",
                    self.session.cwd.name in text
                    and "trust" in text.lower()
                    and bool(re.search(r"1[.)]\s+Yes, (?:continue|proceed)", text)),
                    "1\r",
                ),
            ]
            matched = False
            for label, shown, keys in dialogs:
                if shown:
                    matched = True
                    if label not in handled:
                        self.send(keys, label)
                        handled.add(label)
                    break
            if matched:
                continue
            assert "Select login method:" not in text, (
                "Configured ug launched Claude's account-login flow instead of its gateway session:\n"
                + text
            )
            assert not ("Sign in with ChatGPT" in text and "Provide your own API key" in text), (
                "Configured ug launched Codex's account-login flow instead of its gateway session:\n"
                + text
            )
            title = "Claude Code" if self.agent == "claude" else "Codex"
            if (
                title in text
                and re.search(r"\?\s+for\s+shortcuts", text, re.IGNORECASE)
                and re.search(r"(?m)^\s*[❯›>]", text)
            ):
                self.actions.append({"reason": "prompt-ready", "screen": text})
                return
        raise AssertionError(
            f"TUI did not reach a usable prompt within {timeout}s:\n{self.visible}"
        )

    def check_input_and_exit(self):
        marker = "ug-boot-" + uuid.uuid4().hex[:12]
        self.send(marker, "type into the prompt without submitting a model request")
        self.wait_for(lambda text: marker in text, "typed text in the prompt")
        self.send("\x15", "Ctrl-U clears the prompt")
        self.wait_for(lambda text: marker not in text, "cleared prompt")
        self.send("/exit\r", "exit through the agent's local slash command")
        deadline = time.monotonic() + 30
        while not self.ended and time.monotonic() < deadline:
            self.read()
        assert self.ended, f"TUI did not exit after /exit:\n{self.visible}"
        self.child.close(force=False)
        assert self.child.exitstatus == 0, (
            f"TUI exit={self.child.exitstatus}, signal={self.child.signalstatus}:\n{self.visible}"
        )
