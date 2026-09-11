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
No profile or workspace is selected automatically. The default test selection is `live` (all live CUJs).

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
required; floating `latest`, caret, and tilde versions are rejected. Provider and default-routing CUJs use the workspace's configuration and need no model input. Cases that
exercise explicit model arguments use a real `system.ai` model already discovered
by `ug configure`, recorded in that case's `model.json`. Optional `--claude-model`
and `--codex-model` overrides reproduce a particular model-related failure.

```bash
# Constrain the suspected dependency while keeping the real CLI and gateway.
python3 scripts/run_integration.py \
  --ug-version checkout --claude-version 2.1.268 --codex-version 0.154.0 \
  --claude-model YOUR_CLAUDE_MODEL --codex-model YOUR_CODEX_MODEL \
  --profile YOUR_PROFILE --dependency tomlkit==0.14.0 \
  -- -k 'app_server or app_help'
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

## Test layout and format

All user journeys are top-level tests. There is no separate regressions category:

```text
test_ug_configure_claude.py             # Databricks Hosted and Anthropic MPS
test_ug_configure_codex.py              # Databricks Hosted and OpenAI MPS
test_smart_routing_claude.py            # first prompt, subagent, explicit model
test_smart_routing_codex.py             # first prompt, subagent, explicit model
test_ug_claude_headless.py              # script prompts, models, caller settings
test_ug_codex_headless.py               # script prompts and model arguments
test_ug_claude_commands.py              # command help forwarding
test_ug_codex_commands.py               # command help and parser error forwarding
test_ug_codex_app_server.py             # actual client/server initialize exchange
test_ug_configure_claude_lifecycle.py   # repeat setup, revert, rejected credentials
test_ug_configure_codex_lifecycle.py    # repeat setup, revert, rejected credentials
test_installation.py                   # fresh installed package
utils/                                # process/terminal/evidence helpers and Docker files
```

`conftest.py`, `pytest.ini`, and this README stay at the suite root for pytest
discovery and run instructions.

Each test has a `Scenario:` / `Expected:` docstring and shows its own public
configure command, launch, user action, and assertions. Shared code only handles
process/terminal mechanics, evidence, and cleanup. Fixtures supply an isolated
session and credentials; none manufacture or configure application state.

Provider CUJs use normal ug validation, then require their own completed
interactive task. Routing CUJs skip the preliminary validation prompt and require
the actual routed TUI task instead. Tests disable optional Databricks AI Tools and
pass `--skip-upgrade` to preserve the selected version. They use real onboarding
and trust choices, without seeded acceptance or disabled agent sandboxing.

A fixture file contains an unpredictable value absent from the prompt. Success
requires an assistant answer in the real agent transcript containing that value,
plus normal TUI exit. Codex evidence requires its task-complete event. Subagent
CUJs require a separate child transcript, child answer, and a correlated routing
decision. Claude uses its real child-start audit; its model is unknown when the
event omits it. Codex requires native parent linkage and a completed child turn
using the routed model, excluding inherited parent turns.

MPS CUJs select the existing services already used by e2e:

- Claude: `main.ucode.ci_e2e_anthropic_nonrelay_mps`.
- Codex: `main.ucode.ci_openai_mps`.

Use `--claude-provider` / `--codex-provider` to reproduce another existing service.
Those names are recorded in `versions.json`. No service is created or modified.
A missing service or permission fails the selected CUJ, rather than skipping it.

There are **45 live cases** (including 10 TUI journeys) and **3 installation
checks** with both agents. See the named coverage and gaps matrix in
[../README.md](../README.md).

```bash
# Append one of these selections to the runner command:
-- -m live         # default: all live user journeys
-- -m tui          # ten complete interactive TUI journeys
-- -k test_ug_codex_app_server_client_initializes  # one named journey and its variants
# Use --installation-only before -- for package checks without credentials.
```

The old focused checks are now descriptive CUJs with setup and outcomes visible
in each test. Duplicate boot-only checks are incorporated into the Databricks,
first-prompt, and explicit-model TUI journeys. Real failures, including generated
config left after revert and banners on app-server stdout, remain assertions.
MCP/skills functionality, tracing, the broad configure-option matrix, and other
agents are outside this focused revision.

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

Teardown invokes real `ug revert` through a PTY when setup created state, restoring machine-level
configuration through the public CLI. Per-test homes and working directories are
then deleted even on failure.
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
It runs directly on fresh GitHub Ubuntu VMs, not inside the optional Docker image.
Local native runs use the same runner; Colima/Docker provides a separate Linux
container option. Matching dependency versions does not make those OS environments identical.
Its installation job needs no credentials. For same-repository PRs, the live job
reuse the existing `UCODE_TEST_WORKSPACE` and `DATABRICKS_BEARER` secrets. Fork PRs
run installation checks only because they cannot receive those secrets.

The workspace check requires the secret to match
`https://eng-ml-inference-team-us-east-1.cloud.databricks.com` (a trailing slash
is accepted). It never changes the secret or switches workspaces. There is no CI
model-discovery or model-selection job. Real `ug configure` performs its normal
workspace discovery inside each test; only explicit-model scenarios choose and
record a discovered `system.ai` model as a test argument.
PR CI runs all 45 live cases in one **CUJs** job with fresh consumer dependency
resolution. There is no default dependency matrix. Manual dispatch accepts an
optional `dependency` such as `tomlkit==0.14.0`, equivalent to the local runner's
`--dependency` option. Jobs use Ubuntu 22.04; newer Ubuntu runner
policies prevented Codex's bubblewrap tool from reading even the test file in the
first run. The agent sandbox is not disabled or bypassed.
The workflow consumes the stored bearer; it does not mint or refresh credentials.

For a manual run, use **Actions → Integration → Run workflow**, select the branch,
and choose `live` (default), `tui`, or `installation`. Set the ug/agent versions. From the CLI:

```bash
gh workflow run integration.yml -R databricks/unity-gateway --ref YOUR_BRANCH \
  -f suite=live -f ug_version=checkout \
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
  -n integration-cujs -D .integration-runs/from-ci
```

Use `integration-installation` for package failures. Older runs used numbered
`integration-live-*` artifacts; download the name shown on that run. Read `versions.json` for the
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
  -- -k test_ug_configure_claude_databricks
```

For an explicit-model failure, also pass the recorded `--claude-model` or
`--codex-model`. For a release run without an archived wheel, use the reported `--ug-version`.
Match Python and Node versions from the report too. `npm-lock.json` replay must
use the same OS/architecture as the original run; add `--platform linux/amd64`
to both `docker build` and `docker run` on an ARM Mac to match GitHub's Ubuntu runner. Changing platforms or
resolving a fresh npm lock is a new comparison, not an exact dependency replay.

Use `-- -m tui` for the ten interactive journeys or
`-- -k test_smart_routing_codex_first_prompt` to narrow a failure. Each rerun needs a new output directory. Inspect:

- `junit.xml` for the failing case and assertion.
- `artifacts/<case>/command-*.json` for the real argv, exit status, stdout and stderr.
- `artifacts/<case>/first-session.json`, `provider-session.json`, `first-prompt.json`,
  `subagent-task.json`, or `reopen.json` for rendered terminal screens, raw terminal
  output, keyboard actions, exit status, routing logs, and actual agent-session records.
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
  docker build -f tests/integration/utils/Dockerfile -t ug-integration -

# Reuse the same e2e variables. Credentials are passed at runtime, never built
# into the image. The named volume keeps results after the container exits.
docker volume create ug-integration-results
docker run --rm --init \
  -e UCODE_TEST_WORKSPACE -e DATABRICKS_BEARER \
  -v ug-integration-results:/results \
  ug-integration \
  --ug-version YOUR_RELEASE_VERSION \
  --claude-version 2.1.268 --codex-version 0.154.0 \
  -- -m live
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
