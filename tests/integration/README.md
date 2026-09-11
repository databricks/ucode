# Integration tests

This suite runs the **installed product** through subprocesses, against the same
`UCODE_TEST_WORKSPACE` used by the existing e2e tests. It does not import `ucode`,
patch application functions, substitute agent executables, run a fake gateway,
or construct ug state files. The normal test suite checks these boundaries.

The existing unit tests keep their fixtures. Integration has an independent
pytest configuration and uses `--confcutdir` so those fixtures cannot leak in.
It is not collected by the default `uv run pytest` command.

## Run a specific combination

Prerequisites: Python 3.12+, uv, Node/npm, and Databricks CLI >=1.0.0. The runner
installs the requested agents into a new npm prefix and ug into a new virtualenv.
Pytest and the PTY/screen libraries (pexpect and pyte) live in a different virtualenv, so they cannot accidentally supply a
missing application dependency. No packages are installed into your existing
agent installations or checkout's `.venv`.
Native live runs refuse existing machine-wide Claude/Codex configuration, which
could override the selected workspace even with a fresh home. Use the container
in that case; the runner never edits or bypasses those managed settings.

Use the existing e2e workspace and its `DATABRICKS_BEARER` credential. Locally,
`--profile YOUR_PROFILE` can mint a bearer for an explicitly selected profile.
No profile or workspace is selected automatically.

```bash
export UCODE_TEST_WORKSPACE=https://your-existing-e2e-workspace

python3 scripts/run_integration.py \
  --ug-version checkout \
  --claude-version 2.1.268 \
  --codex-version 0.154.0 \
  --profile YOUR_PROFILE
```

`checkout` builds a wheel and installs it with fresh consumer dependency
resolution. **It does not use `uv.lock`.** This exercises the install path that
caught the tomlkit discrepancy in #496. To reproduce a user's release, pass its
exact distribution version instead, e.g. `--ug-version 0.1.0+f7b4b97`. Use
`--index-url` for the Python index that contains that release and `--npm-registry`
for an npm mirror if public npm is unavailable. Older releases that only
provide the `ucode` command require `--entry-point ucode`.

Select one agent by providing only its version. Exact agent versions are
required; floating `latest`, caret, and tilde versions are rejected. Normal TUI
boot uses the workspace's configuration and needs no model input. Cases that
exercise explicit model arguments use a real `system.ai` model already discovered
by `ug configure`, recorded in that case's `model.json`. Optional `--claude-model`
and `--codex-model` overrides reproduce a particular model-related failure.

```bash
# Constrain the suspected dependency while keeping the real CLI and gateway.
python3 scripts/run_integration.py \
  --ug-version checkout --claude-version 2.1.268 --codex-version 0.154.0 \
  --claude-model YOUR_CLAUDE_MODEL --codex-model YOUR_CODEX_MODEL \
  --profile YOUR_PROFILE --dependency tomlkit==0.14.0 \
  -- -k 'app_server or subcommand'
```

Repeat with `--dependency tomlkit==0.15.1`, or run without constraints to test
what a new consumer gets today. Several `--dependency` options can be supplied.
Constraints incompatible with the selected ug release fail installation.

For package-only validation without credentials:

```bash
python3 scripts/run_integration.py \
  --ug-version checkout --claude-version 2.1.268 --installation-only
```

This explicitly selects only the installation checks; it does not claim a live
integration pass. Requested live checks fail when credentials, binaries, models,
or capabilities are missing. There are no capability-based skips or retries of
failed model tasks. A failing historical version should remain a failing result.

## Coverage and boundaries

| Area | Automated evidence |
| --- | --- |
| Installed package | Console entry point, version, clean-home status, and missing-configuration error from site-packages |
| Configuration | CLI discovery against the real workspace, repeat setup, preserved user settings, revert and repeat revert |
| Authentication failure | A deliberately invalid bearer is rejected by the real workspace before setup succeeds |
| Codex utility dispatch | Real `app`, `app-server`, `exec`, and `mcp` subcommand help with routing on/off, compared with the selected Codex binary's own output |
| Direct Codex app arguments | An invalid option must reach the real `codex app` parser and preserve its error/exit status, without starting a desktop app |
| Claude utility dispatch | Real `auth` and `mcp` subcommand help with routing on/off |
| Codex app server | A real stdio `initialize` exchange with routing on/off, with and without a launcher `--` separator; JSON must arrive on stdout, separately from stderr diagnostics |
| Agent execution | A real agent reads an unpredictable value from a fixture file and returns it in its structured final response |
| Model options | `--model VALUE`, `--model=VALUE`, and `-m VALUE`, forwarded after `--`, with global routing enabled; none may start a routing wrapper |
| Prompt input | Argument, stdin, and nested `--` forms must complete the same real file-reading task |
| Caller settings | Claude receives a settings path containing spaces, runs the caller's hook, and still authenticates through ug |
| Interactive boot | Real PTY; first agent startup and reopening the same home; routing on/off and explicit-model bypass; visible onboarding, prompt keyboard input, `/exit` and exit status 0 |
| Dependency compatibility | Independent consumer resolution plus explicit constraints; CI exercises tomlkit 0.14.0 and 0.15.1 |

With both agents selected, there are 41 cases: 3 installation, 4 lifecycle,
18 utility/protocol, 10 headless task cases, and 6 TUI cases. Each TUI case
boots twice (first startup and reopen). The `--` and caller-settings
cases exercise the argument forms used by launchers
such as Isaac. **They do not launch Isaac itself.** Interactive first-prompt
routing, actual desktop app startup, terminal resize/signals, OS-managed settings,
and updater execution are not covered by the current tests. They require
native/PTY scenarios, especially on macOS; `codex app --help` proves dispatch and
config serialization, not that the desktop app opened. New regressions should
add an explicit row/case rather than broaden the meaning of an existing test.

**TUI fidelity:** `test_tui.py` starts installed `ug claude` / `ug codex` in a real
controlling terminal. After the CLI creates gateway configuration, it walks
through recognized visible agent onboarding/trust screens, requires the TUI's
prompt, types and clears text, exits through `/exit`, and repeats with the state
the agent actually wrote. It also checks that routing wrappers start only in the
expected launch modes and do not route before a model prompt. Unknown onboarding
screens, missing prompts, abnormal exits, and timeouts fail with terminal evidence.
No onboarding state is fabricated. The terminal libraries render ANSI output and
answer terminal-device queries; they do not emulate agent or gateway behavior.

These boot cases do **not** submit inference requests, cover a second conversation
turn, or exercise tool permission dialogs. Real task execution is separately
tested using Claude `-p` and Codex `exec`. The next TUI cases need a completed
interactive task and successful first-prompt routing. Colima isolates the Linux
environment; native macOS/Windows behavior needs its own runs.

Run just the TUI cases by adding `-- -m tui` to the runner command.

Live task tests make inference requests; model overrides can bound their cost.
The remote gateway, its managed settings, and model availability remain external
inputs; this suite is isolated, not an offline emulation of Databricks.

## Reproduce a failure

Each run writes a new `.integration-runs/<timestamp>/` directory containing:

- `versions.json`: requested and observed ug/agent versions, Python, Node, uv,
  Databricks CLI, platform, source revision/diff, suite hash, and wheel hash when available.
- `dependencies.txt` and `npm-lock.json`: the resolved Python and npm dependency
  graphs. Replay them with `--constraints` and `--npm-lock`.
- `test-dependencies.txt`: the separately installed pytest/terminal-tool dependencies.
- `junit.xml`: exact test outcomes and parametrized case names.
- `artifacts/`: command arguments, exit codes, timeout status, redacted output,
  app-server protocol diagnostics, and TUI transcripts/rendered screens plus
  keystroke actions and routing logs. No credential files are archived.
- `wheels/`: the tested wheel when built from the checkout; replay it with
  `--ug-wheel`. For release installations, `installed.txt` records the resolution.

Per-test homes and working directories are deleted even on failure.
The working directory is outside the checkout so an agent cannot inherit its
project settings or instruction files by walking parent directories. Virtualenvs,
agent packages, and build caches remain under the results directory for local
inspection; remove that run directory when finished. Agent versions are checked
before and after the suite so an automatic upgrade cannot silently change the
combination being tested. Model requests and subprocesses have deadlines, and
the process group is cleaned up after each command.
Selection after `--` accepts `-k`, `-m`, `-x`, and `--maxfail`; configuration and
report paths cannot be overridden. `--installation-only` always restricts the
selection to installation checks, including when additional filters are used.

## Run in GitHub Actions

The **Integration** workflow runs on relevant pull requests and pushes to `main`.
Its installation job needs no credentials. For same-repository PRs, the live jobs
reuse the existing `UCODE_TEST_WORKSPACE` and `DATABRICKS_BEARER` secrets. Fork PRs
run installation checks only because they cannot receive those secrets.

The workspace check requires the secret to match
`https://eng-ml-inference-team-us-east-1.cloud.databricks.com` (a trailing slash
is accepted). It never changes the secret or switches workspaces. There is no CI
model-discovery or model-selection job. Real `ug configure` performs its normal
workspace discovery inside each test; only explicit-model scenarios choose and
record a discovered `system.ai` model as a test argument.
The live matrix covers unconstrained resolution, tomlkit 0.14.0, and tomlkit 0.15.1.
The workflow consumes the stored bearer; it does not mint or refresh credentials.

For a manual run, use **Actions → Integration → Run workflow**, select the branch,
and choose `all`, `tui`, or `installation`. Set the ug/agent versions. From the CLI:

```bash
gh workflow run integration.yml -R databricks/unity-gateway --ref YOUR_BRANCH \
  -f suite=tui -f ug_version=checkout \
  -f claude_version=2.1.268 -f codex_version=0.154.0
gh run list -R databricks/unity-gateway --workflow integration.yml
gh run watch RUN_ID -R databricks/unity-gateway --exit-status
```

GitHub enables manual dispatch once the workflow exists on the default branch.
Before this PR merges, its pull-request event runs the workflow. Missing
credentials or a workspace mismatch fail the workspace job. Expired or invalid
credentials fail the actual workspace calls. Those failures do not count as live
test passes.

## Reproduce and debug a CI failure locally

Use the same runner and the failing job's artifacts. A new developer machine
needs Python 3.12+, uv, Node/npm, Databricks CLI, and its own authorized login for
the CI workspace. Select that local profile explicitly; CI secrets are not downloaded.

```bash
gh run download RUN_ID -R databricks/unity-gateway \
  -n integration-live-0 -D .integration-runs/from-ci
```

Use `integration-live-1` or `integration-live-2` for the other dependency jobs,
or `integration-installation` for package failures. Read `versions.json` for the
exact agent versions, model overrides, entry point, platform and source revision.
For an explicit-model case without a runner override, read its `model.json` for
the exact model used. Basic boot cases require no model arguments. Use
the archived wheel so a changed checkout cannot alter the reproduction:

```bash
python3 scripts/run_integration.py \
  --ug-wheel .integration-runs/from-ci/wheels/EXACT_WHEEL.whl \
  --entry-point ug \
  --claude-version CLAUDE_VERSION_FROM_REPORT --codex-version CODEX_VERSION_FROM_REPORT \
  --workspace https://eng-ml-inference-team-us-east-1.cloud.databricks.com \
  --profile YOUR_E2E_PROFILE \
  --constraints .integration-runs/from-ci/dependencies.txt \
  --npm-lock .integration-runs/from-ci/npm-lock.json \
  --output .integration-runs/repro-1 \
  -- -k test_tui_boot_reopen_and_exit
```

For an explicit-model failure, also pass the recorded `--claude-model` or
`--codex-model`. For a release run without an archived wheel, use the reported `--ug-version`.
Match Python and Node versions from the report too. `npm-lock.json` replay must
use the same OS/architecture as the original run; add `--platform linux/amd64`
to both `docker build` and `docker run` on an ARM Mac to match GitHub's Ubuntu runner. Changing platforms or
resolving a fresh npm lock is a new comparison, not an exact dependency replay.

Use `-- -m tui` for all boot cases or `-- -k 'codex and routing-on'` to narrow a
failure. Each rerun needs a new output directory. Inspect:

- `junit.xml` for the failing case and assertion.
- `artifacts/<case>/command-*.json` for the real argv, exit status, stdout and stderr.
- `artifacts/<case>/first-boot.json` / `reopen.json` for rendered terminal screens, raw terminal
  output, keyboard actions, exit status and routing logs.
- `install.log` for resolution/bootstrap failures.

Unknown onboarding screens fail with their actual screen text. Update terminal
selectors only after confirming the agent's intended UI changed; do not seed its
onboarding state or relax the prompt/task assertions. Test homes are deleted after
each case; redacted diagnostics remain. For manual interaction, configure a fresh
home with the same installed binaries and recorded public CLI arguments.

## Colima / Docker

Colima provides the Linux Docker engine on macOS. The optional image pins the
Python, Node, uv, and Databricks toolchain; the same runner selects ug and agent
versions inside it. Build from the repository root:

```bash
colima start
COPYFILE_DISABLE=1 tar --format=ustar --exclude=__pycache__ --exclude=.pytest_cache \
  -cf - scripts/run_integration.py tests/integration | \
  docker build -f tests/integration/Dockerfile -t ug-integration -

# Reuse the same e2e variables. Credentials are passed at runtime, never built
# into the image. The named volume keeps results after the container exits.
docker volume create ug-integration-results
docker run --rm --init \
  -e UCODE_TEST_WORKSPACE -e DATABRICKS_BEARER \
  -v ug-integration-results:/results \
  ug-integration \
  --ug-version YOUR_RELEASE_VERSION \
  --claude-version 2.1.268 --codex-version 0.154.0 \
  -- -m tui
```

Use a new results volume for each run, or pass a new `--output /results/NAME`.
To test a checkout, build a wheel on the host (`uv build --wheel`), mount the
wheel directory read-only, and pass `--ug-wheel /wheels/FILE.whl` instead of a
release version. The image deliberately contains no source checkout or host
agent configuration. Record the built image digest when sharing a reproduction;
native runs also depend on the host's OS and toolchain.
The explicit build archive includes only the runner and integration files, even
with legacy Docker builders that ignore per-Dockerfile ignore rules. It also
omits macOS extended attributes that Linux cannot unpack.
