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

from .evidence import agent_sessions


class TerminalScreen(pyte.Screen):
    def __init__(self, columns, lines, send):
        super().__init__(columns, lines)
        self.send = send

    def write_process_input(self, data):
        # Real TUIs query cursor position/device attributes during startup.
        # pyte replies according to the terminal state it has actually rendered.
        self.send(data)


class TerminalProcess:
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
            dimensions=(60, 140),
            timeout=120,
        )
        self.screen = TerminalScreen(140, 60, self.child.send)
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
                "terminal": {"rows": 60, "columns": 140, "term": "xterm-256color"},
                "actions": self.actions,
                "transcript": "".join(self.output),
                "screen": self.visible,
                "exitstatus": self.child.exitstatus,
                "signalstatus": self.child.signalstatus,
                "normal_exit_observed": self.ended and self.child.exitstatus == 0,
                "routing_log": routing_log.read_text() if routing_log.is_file() else None,
                "agent_sessions": agent_sessions(self.session, self.agent)
                if self.agent in ("claude", "codex")
                else {},
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

    def wait_for(self, predicate, description, timeout=30, stable_for=0.3):
        deadline = time.monotonic() + timeout
        since = None
        while time.monotonic() < deadline:
            self.read()
            assert not self.ended, f"TUI exited while waiting for {description}:\n{self.visible}"
            if predicate(self.visible):
                since = since or time.monotonic()
                if time.monotonic() - since >= stable_for:
                    return
            else:
                since = None
        raise AssertionError(f"TUI did not show {description} within {timeout}s:\n{self.visible}")

    def selected_line(self):
        return next(
            (line.strip() for line in self.visible.splitlines() if re.match(r"^\s*[›❯>]", line)),
            "",
        )

    def choose(self, prompt, label):
        """Navigate the visible menu with arrow keys; never write its saved state."""
        self.wait_for(lambda text: prompt in text and self.selected_line(), prompt, timeout=120)
        visited = set()
        for _ in range(100):
            current = self.selected_line()
            if label in current:
                self.send("\r", f"choose {label}")
                self.wait_for(
                    lambda text, before=current: (
                        prompt not in text or self.selected_line() != before
                    ),
                    f"confirmation of {label}",
                )
                return
            assert current not in visited, f"Menu does not offer {label}:\n{self.visible}"
            visited.add(current)
            self.send("\x1b[B", f"move towards {label}")
            self.wait_for(
                lambda text, before=current: self.selected_line() != before, "next menu option"
            )
        raise AssertionError(f"Could not select {label}")

    def submit(self, text):
        self.send(text, "type text before pressing Enter")
        compact = "".join(text.split())
        self.wait_for(
            lambda screen: compact in "".join(screen.split()), "typed input", stable_for=0.5
        )
        self.send("\r", "press Enter after the input has rendered")

    def finish(self, timeout=120):
        deadline = time.monotonic() + timeout
        while not self.ended and time.monotonic() < deadline:
            self.read()
        assert self.ended, f"Process did not exit within {timeout}s:\n{self.visible}"
        self.child.close(force=False)
        assert self.child.exitstatus == 0, (
            f"exit={self.child.exitstatus}, signal={self.child.signalstatus}:\n{self.visible}"
        )


class ConfigureTerminal(TerminalProcess):
    def select_agent(self, display):
        prompt = "Select coding agents to configure:"
        self.wait_for(lambda text: prompt in text and self.selected_line(), prompt, timeout=120)
        visited = set()
        selected = False
        while True:
            current = self.selected_line()
            match = re.search(r"[›❯>]\s*([●○])\s*(.+)", current)
            assert match, f"Unrecognized agent checkbox: {current}"
            checked, name = match.groups()
            if name in visited:
                break
            visited.add(name)
            wanted = name.strip() == display
            selected = selected or wanted
            if (checked == "●") != wanted:
                self.send(" ", f"{'select' if wanted else 'deselect'} {name}")
                self.wait_for(
                    lambda text, before=current: self.selected_line() != before, "checkbox change"
                )
            current = self.selected_line()
            self.send("\x1b[B", "next agent checkbox")
            self.wait_for(lambda text, before=current: self.selected_line() != before, "next agent")
        assert selected, f"{display} was not available in the real agent picker"
        self.send("\r", f"configure only {display}")


class AgentTerminal(TerminalProcess):
    def boot(self, timeout=120):
        """Handle only recognized visible onboarding; unknown screens fail."""
        handled = set()
        ready_since = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.read()
            assert not self.ended, f"TUI exited before its prompt:\n{self.visible}"
            text = self.visible
            if "Accessing workspace:" in text and self.session.cwd.name in text:
                self.choose("Accessing workspace:", "Yes, I trust this folder")
                continue
            if "Hooks need review" in text:
                self.choose("Hooks need review", "Trust all and continue")
                continue
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
                    "\r",
                ),
                (
                    "trust-directory",
                    self.session.cwd.name in text
                    and "trust" in text.lower()
                    and bool(re.search(r"1[.)]\s+Yes, (?:continue|proceed)", text)),
                    "\r",
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
                and "loading" not in text.lower()
                and re.search(r"(?m)^\s*[❯›>]\s*(?!\d+[.)])", text)
            ):
                ready_since = ready_since or time.monotonic()
                if time.monotonic() - ready_since >= 1:
                    self.actions.append({"reason": "prompt-ready", "screen": text})
                    return
            else:
                ready_since = None
        raise AssertionError(
            f"TUI did not reach a usable prompt within {timeout}s:\n{self.visible}"
        )

    def check_input_and_exit(self):
        marker = "ug-boot-" + uuid.uuid4().hex[:12]
        self.send(marker, "type into the prompt without submitting a model request")
        self.wait_for(lambda text: marker in text, "typed text in the prompt")
        self.send("\x15", "Ctrl-U clears the prompt")
        self.wait_for(lambda text: marker not in text, "cleared prompt")
        self.exit_normally()

    def wait_for_task(self, task, timeout=180):
        self.wait_for(
            lambda text: task.completed(self.session, self.agent),
            "a completed assistant answer with the file's value",
            timeout=timeout,
        )
        task.assert_completed(self.session, self.agent)

    def exit_normally(self):
        self.submit("/exit")
        self.finish(timeout=30)
