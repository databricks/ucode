# Test suites and user journeys

The integration suite runs a freshly installed ug wheel/release, exact real
Claude/Codex versions, and the existing real e2e workspace. It has no application
imports, mocks, monkeypatching, fake binaries/services, or fabricated ug state.

| Category | Location | What it proves |
| --- | --- | --- |
| Unit/component | Existing `test_*.py` files | Individual behavior; dependencies may be mocked |
| Existing e2e | `test_e2e*.py` | Real workspace behavior with some patched setup/internal calls |
| Main integration CUJs | `integration/test_ug_configure_*.py`, `integration/test_smart_routing_*.py` | Complete configure → real TUI → task → exit journeys |
| Installation | `integration/test_installation.py` | Fresh installed package without credentials |
| Focused integration regressions | `integration/regressions/` | Public command, argument, protocol, and lifecycle contracts |

## Main CUJ matrix

These are **implemented assertions**, not a claim that every agent/version
combination passes. Consult the run's JUnit report and artifacts for results.
Each function states its **Scenario** and **Expected** outcome and shows its
configure and launch commands. No fixture silently configures the application.

| Test | Setup and user action | Expected evidence |
| --- | --- | --- |
| `test_ug_configure_claude_databricks` | Configure Claude with Databricks Hosted; open TUI and read a file | Assistant returns an unpredictable file value, exits normally, and reopens with working keyboard input |
| `test_ug_configure_claude_anthropic_mps` | Select the existing Anthropic MPS in the real configure picker; launch without a provider override | Saved provider appears in status; real Claude completes the file task and exits |
| `test_ug_configure_codex_databricks` | Configure Codex with Databricks Hosted; open TUI and read a file | Completed assistant answer contains the file value, normal exit and reopen |
| `test_ug_configure_codex_openai_mps` | Select the existing OpenAI MPS in the real configure picker; launch without a provider override | Saved provider appears in status; real Codex completes the file task and exits |
| `test_smart_routing_claude_first_prompt` | Configure, enable routing, type the first TUI prompt | Real routing decision and prompt replay plus completed task; no routing before submission |
| `test_smart_routing_codex_first_prompt` | Configure, enable routing, type the first TUI prompt | Real routing decision plus completed task; no fallback or routing before submission |
| `test_smart_routing_claude_subagent` | Ask Claude to delegate a file-reading task | Actual child transcript with the answer, correlated routing decision/child start, parent answer |
| `test_smart_routing_codex_subagent` | Ask Codex to delegate a file-reading task | Actual child session with the answer, correlated routing decision/child start, parent answer |

The main configuration tests include ug's normal validation. Routing tests use
`--skip-validate` during setup because their own TUI task is the validation.
All keep the requested agent versions with `--skip-upgrade` and disable optional
Databricks AI Tools to keep these basic journeys focused.

## Retained regression coverage

The original focused checks moved into `integration/regressions/`; they were not
deleted when the main suite was narrowed. Select them with `-- -m regression`.

| Concern | Coverage |
| --- | --- |
| Fresh consumer resolution differs from `uv.lock` (#496) | Fresh wheel install; dependency constraints/replay; CI tests unconstrained, tomlkit 0.14.0 and 0.15.1 |
| Reconfigure, preserve user settings, revert | CLI-created state and real files; known generated-file cleanup failure remains an assertion |
| Rejected credentials | Real workspace rejection and no successful saved setup |
| Subcommand forwarding (#502) | Real Claude auth/mcp and Codex app/app-server/exec/mcp help, routing off/on |
| Codex app argument error | Real parser error and exit status; excludes unrelated per-launch warnings |
| Codex app-server | Real JSON-RPC initialize, direct/separator forms; non-JSON stdout still fails |
| Headless prompt/model arguments | Real file task, stdin/separators and caller settings/hook; Claude's unsupported `-m` must retain its real error |
| TUI boot modes | Routing off/on/explicit-model, first boot/reopen, input/clear/exit; no inference claim |

## Gaps and deferred scope

| Scenario | Status / requirement |
| --- | --- |
| MCP and skills CUJs | Deferred at the user's request |
| Broad configure flags, tracing, multiple workspaces, OAuth/PAT flows | Deferred while focusing on basic main CUJs |
| Provider switching, relayed/subscription MPS | Not covered by the four basic provider journeys |
| Initial prompt supplied on the launch command line | Not yet covered by main routing CUJs |
| Follow-up turns and conversation resume | Not covered; reopen proves startup, not conversation resume |
| Exact child model identity | Verified only when the real agent reports it; missing model fields remain unknown in artifacts |
| Full allow/deny tool-permission matrix | Not covered; real onboarding/trust choices are handled through the TUI |
| Desktop Codex app, Isaac itself, auto-upgrades | Not covered by command forwarding or pinned-version tests |
| Native macOS/Windows managed settings, resize/signals | Separate platform coverage needed |
| Other agents | Current main scope is Claude Code and Codex |

See [integration/README.md](integration/README.md) for commands, CI, artifacts,
and reproduction. Follow [AGENTS.md](AGENTS.md) and [CLAUDE.md](CLAUDE.md) when
adding, modifying, or removing tests. The ordinary suite enforces both the
no-mocking boundary and the Scenario/Expected docstring format.
