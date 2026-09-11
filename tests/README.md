# Test suites and coverage

The categories prove different things. A passing unit test or existing e2e test
does not imply that the installed CLI's complete workflow was exercised.

| Category | Location | What is real? | What can be substituted? | Run |
| --- | --- | --- | --- | --- |
| Unit / component | `test_*.py`, excluding the suites below | Function or component under test | Dependencies, subprocesses, network and state paths | `uv run pytest` |
| Existing e2e | `test_e2e.py`, `test_e2e_uc.py`, `test_e2e_tracing.py` | Databricks workspace; some tests launch agents | Several tests patch state/configuration or invoke application helpers directly | `UCODE_TEST_WORKSPACE=… uv run pytest tests/test_e2e.py -v` |
| Existing proxy / wire tests | `test_gateway_proxy_integration.py`, `test_e2e_user_agent.py` | Local HTTP sockets; real agents in User-Agent tests | Local upstream responses, token minting, config/state paths | `uv run pytest tests/test_gateway_proxy_integration.py -v` |
| **Integration** | **`integration/`** | **Installed ug wheel/release, selected real agents, real e2e workspace, real files/processes** | **No application or service behavior: no mocks, monkeypatching, fake servers, or fabricated ug state** | **`python3 scripts/run_integration.py …`** |

Integration has its own pytest configuration and does not inherit the unit
suite's state-patching fixture. Its runner installs ug and pytest into different
environments and does not consume the checkout's `uv.lock`.

## New integration coverage matrix

**Covered** means an automated assertion exists, not that every version
combination has passed. Consult a run's `junit.xml` and `versions.json` for actual
results. **Not covered** is a gap, not a promise made by a neighboring test.

| Behavior / concern | Claude Code | Codex | Test / limitation |
| --- | --- | --- | --- |
| Build/install checkout wheel | Covered | Covered | Runner + `test_installation.py`; imports resolve inside installed runtime |
| User's exact ug and agent versions | Covered | Covered | Release/wheel and exact agent versions; verified before/after execution |
| Consumer dependencies differ from `uv.lock` (#496) | Covered | Covered | Fresh resolution; constraints; CI matrix includes tomlkit 0.14.0 and 0.15.1 |
| Clean-home help, version, status, auth guidance | Covered | Covered | `test_installation.py`; no workspace needed for these checks |
| Configure against real e2e workspace | Covered | Covered | `test_lifecycle.py`; state is produced only by CLI commands |
| Repeat setup; preserve unrelated user settings | Covered | Covered | `test_lifecycle.py` |
| Revert and repeat revert | Covered | Covered | `test_lifecycle.py` |
| Invalid credentials rejected by real service | Covered | Covered | `test_lifecycle.py`; nonzero exit, no successful saved setup |
| Utility dispatch, routing off/on (#502) | `auth`, `mcp` | `app`, `app-server`, `exec`, `mcp` | `test_passthrough.py`; real subcommand help, compared with direct agent invocation |
| Direct `codex app` argument forwarding | Not applicable | Covered | An invalid option must produce the real agent's error and exit status, without opening a desktop app |
| App-server initialization protocol | Not applicable | Covered | Real JSON-RPC on stdout, separate stderr; direct/launcher `--` forms, routing off/on; rejects non-JSON protocol output |
| Agent reads file through real gateway | Covered | Covered | `test_tasks.py`; unpredictable fixture value in structured final answer |
| Explicit model bypasses global routing | Covered | Covered | `--model VALUE`, `--model=VALUE`, and `-m VALUE`; no routing wrapper diagnostics |
| Stdin prompts and nested `--` separators | Covered | Covered | Real file-reading task with prompt piped on stdin or supplied after the agent's own separator |
| Launcher options after `--` | Covered | Covered | Task and app-server cases |
| Caller settings, path with spaces, preserved hook | Covered | Not covered | Claude caller hook executes while gateway authentication still works |
| Interactive TUI boot and reopen after `ug configure` | Covered | Covered | `test_tui.py`; real PTY, first agent startup and persisted-home reopen; routing on/off and explicit-model bypass |
| First ug launch with automatic configuration/upgrades | **Not covered** | **Not covered** | Boot tests explicitly configure ug first; automatic upgrades need a separate version-change scenario |
| TUI prompt keyboard input and normal exit | Covered | Covered | Type and clear an unsubmitted prompt, execute `/exit`, require exit 0; terminal transcripts and rendered screens |
| TUI first-prompt inference / initial prompt after `--` | **Not covered** | **Not covered** | Boot cases submit no model prompt; needs completed interactive tasks and successful first-prompt routing |
| Interactive follow-up prompts and conversation state | **Not covered** | **Not covered** | Headless single-turn tasks do not exercise the TUI's next turn |
| TUI onboarding and project trust | Covered | Covered | Recognized visible dialogs are handled through keystrokes; no seeded onboarding state; unknown screens fail |
| TUI tool permission dialogs | **Not covered** | **Not covered** | Boot cases do not invoke tools or exercise allow/deny decisions |
| Smart routing selects a model | **Not covered** | **Not covered** | Current tests establish when routing must be bypassed |
| Desktop application startup | Not applicable | **Not covered** | `app --help` checks dispatch/serialization, not desktop startup |
| Agent updater execution | **Not covered** | **Not covered** | Needs a separate scenario that intentionally changes versions |
| Isaac starts and completes a session | **Not covered** | **Not covered** | Launcher-style boundaries are tested; Isaac executable is not |
| Native macOS/Windows UI, OS-managed settings, signals/resize | **Not covered** | **Not covered** | Linux container checks do not establish native-platform behavior |
| Workspace switching, token expiry, MCP, skills, tracing | **Not covered here** | **Not covered here** | Existing tests cover some components; no complete new integration journey |
| Gemini, OpenCode, Copilot, Pi, Cursor | **Not covered here** | **Not covered here** | Initial integration scope is Claude and Codex |

See [integration/README.md](integration/README.md) for version selection,
dependency replay, Colima/Docker, CI, and report contents.

## Maintaining tests

Read [AGENTS.md](AGENTS.md) before adding, changing, or removing tests. Keep this
matrix aligned with actual assertions, including gaps. A regression should name
the broken user command and version combination and exercise the real installed
program. Never replace a broken integration path with an internal function call
or a successful canned response.

The ordinary suite includes `test_integration_contract.py`, which rejects
application imports and common mocking/patching constructs in `integration/`.
It is a guardrail, not a substitute for reviewing what a test actually proves.
