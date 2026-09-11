# Test suites and user journeys

Integration runs a freshly installed ug wheel/release, exact real Claude/Codex
versions, and the existing real e2e workspace. It has no application imports,
mocks, monkeypatching, fake binaries/services, or fabricated ug state.

| Category | Location | What it proves |
| --- | --- | --- |
| Unit/component | Existing `test_*.py` files | Individual behavior; dependencies may be mocked |
| Existing e2e | `test_e2e*.py` | Real workspace behavior with some patched setup/internal calls |
| Integration CUJs | `integration/test_*.py` | Public configure, TUI, script, command, protocol, and lifecycle journeys |
| Installation | `integration/test_installation.py` | Fresh installed package without credentials |

## CUJ coverage matrix

These are **implemented assertions**, not a claim that every version passes.
Consult the run's JUnit report and artifacts for results. Each function states
its **Scenario** and **Expected** outcome and shows its configure and launch
commands. Fixtures supply fresh environments and credentials, never configured ug.
All tests live directly in `integration/`; shared mechanics live in `utils/`.

| Test | User action | Expected evidence |
| --- | --- | --- |
| `test_ug_configure_claude_databricks` | Configure Databricks Hosted; open Claude TUI and read a file | Assistant returns an unpredictable file value; normal exit; reopen with working keyboard input |
| `test_ug_configure_claude_anthropic_mps` | Select Anthropic MPS in the real configure picker; launch Claude | Saved provider in status; completed TUI file task; normal exit |
| `test_ug_configure_codex_databricks` | Configure Databricks Hosted; open Codex TUI and read a file | Completed assistant answer contains the file value; normal exit and reopen |
| `test_ug_configure_codex_openai_mps` | Select OpenAI MPS in the real configure picker; launch Codex | Saved provider in status; completed TUI file task; normal exit |
| `test_smart_routing_claude_first_prompt` | Enable routing and type the first TUI prompt | Real routing decision and replay; completed task; no routing before submission; reopen |
| `test_smart_routing_codex_first_prompt` | Enable routing and type the first TUI prompt | Real routing decision; completed task; no fallback or routing before submission; reopen |
| `test_smart_routing_claude_subagent` | Delegate a file-reading task | Child answer, correlated routing decision/child start, parent answer |
| `test_smart_routing_codex_subagent` | Delegate a file-reading task | Native child-parent linkage; completed child turn uses the routed model; parent answer |
| `test_smart_routing_claude_explicit_model_bypasses_routing` | Launch TUI with an explicit model and routing enabled | Completed task, no routing wrapper, normal exit and reopen |
| `test_smart_routing_codex_explicit_model_bypasses_routing` | Launch TUI with an explicit model and routing enabled | Completed task, no routing wrapper, normal exit and reopen |
| `test_ug_claude_headless_prompt_argument`, `test_ug_claude_headless_prompt_stdin`, `test_ug_claude_headless_prompt_after_separator` | Run Claude from a script using each prompt form | Structured final answer contains the file value; exit zero; no routing |
| `test_ug_codex_headless_prompt_argument`, `test_ug_codex_headless_prompt_stdin`, `test_ug_codex_headless_prompt_after_separator` | Run Codex from a script using each prompt form | Completed turn and final answer contain the file value; exit zero; no routing |
| `test_ug_claude_headless_explicit_model_bypasses_routing` | Pass `--model VALUE` / `--model=VALUE` with routing enabled | Real file task completes; no routing wrapper |
| `test_ug_codex_headless_explicit_model_bypasses_routing` | Pass `--model VALUE` / `--model=VALUE` / `-m VALUE` with routing enabled | Real file task completes; no routing wrapper |
| `test_ug_claude_preserves_caller_settings_and_hook` | Pass a settings path containing spaces | Real SessionStart hook executes; caller file unchanged; file task completes |
| `test_ug_claude_reports_unsupported_short_model_option` | Pass Claude's unsupported `-m` | Actual agent error and exit status preserved |
| `test_ug_claude_auth_help`, `test_ug_claude_mcp_help` | Request subcommand help, routing off/on | Real agent help; no routing wrapper |
| `test_ug_codex_app_help`, `test_ug_codex_app_server_help`, `test_ug_codex_exec_help`, `test_ug_codex_mcp_help` | Request subcommand help, routing off/on | Real agent help; no routing wrapper |
| `test_ug_codex_app_reports_unknown_argument` | Pass an invalid option directly to `ug codex app`, routing off/on | Real Codex parser error and status preserved |
| `test_ug_codex_app_server_client_initializes` | Connect a stdio client, direct/`--` separator, routing off/on | Actual JSON-RPC initialize response; no non-JSON stdout; no routing |
| `test_ug_configure_claude_repeat_and_revert`, `test_ug_configure_codex_repeat_and_revert` | Configure twice over user settings; complete a task; revert twice | Settings preserved; no bearer in ug state; generated config removed; status unconfigured |
| `test_ug_configure_claude_rejects_invalid_credentials`, `test_ug_configure_codex_rejects_invalid_credentials` | Configure with a rejected bearer against the real workspace | Authentication failure; no successful saved setup |
| `test_ug_installed_wheel_exposes_help_and_version` | Invoke freshly installed console command | Package version matches; public help works |
| `test_ug_status_in_fresh_home_is_unconfigured` | Request status before configure | Unconfigured status |
| `test_ug_auth_without_configuration_explains_how_to_configure` | Request auth before configure | Actionable setup error and nonzero exit |

With both agents selected there are **45 live cases** (10 interactive TUI cases)
and **3 installation checks**. Parametrization varies argument spelling or routing
mode, never hides the agent/provider in the test name. Duplicate boot-only cases
were merged into the Databricks, first-prompt, and explicit-model TUI journeys.
Generated-file cleanup and strict app-server stdout assertions remain enforced.

Provider configuration tests include ug's normal validation. Other setup uses
`--skip-validate` when the journey supplies its own task or checks a command
contract. Tests use `--skip-upgrade` to preserve selected agent versions and disable
optional Databricks AI Tools. Help forwarding does not claim MCP functionality.

Fresh consumer dependency resolution covers the install path behind #496, rather
than consuming `uv.lock`. Use `--dependency PACKAGE==VERSION` or replay the archived
dependency graph to reproduce a user's combination. PR CI runs one live CUJ job.

## Gaps and deferred scope

| Scenario | Status / requirement |
| --- | --- |
| MCP and skills functionality | Deferred at the user's request; existing `mcp --help` dispatch checks only |
| Broad configure flags, tracing, multiple workspaces, OAuth/PAT flows | Deferred while focusing on basic CUJs |
| Provider switching, relayed/subscription MPS | Not covered by the four provider journeys |
| TUI initial prompt supplied on the launch command line | Not yet covered; headless prompt arguments are covered |
| Follow-up turns and conversation resume | Not covered; reopen proves startup, not conversation resume |
| Claude child model identity | Verified only when the actual start event reports it; missing fields remain unknown |
| Full allow/deny tool-permission matrix | Not covered; onboarding/trust uses actual TUI choices |
| Desktop Codex app, Isaac itself, auto-upgrades | Not covered by command forwarding or pinned-version tests |
| Native macOS/Windows managed settings, resize/signals | Separate platform coverage needed |
| Other agents | Current scope is Claude Code and Codex |

See [integration/README.md](integration/README.md) for commands, CI, artifacts,
and reproduction. Follow [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) when
adding, modifying, or removing tests. The ordinary suite enforces both the
no-mocking boundary and the Scenario/Expected docstring format.
