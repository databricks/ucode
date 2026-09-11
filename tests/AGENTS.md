# Instructions for agents editing tests

Read `README.md` and `integration/README.md` before adding, removing, or changing
tests. Keep work scoped to the behavior requested by the user.

## Categories

- Existing unit/component tests may use focused mocks. Do not move their global
  fixtures into integration or rewrite them all as part of a narrow fix.
- `integration/` tests installed ug with real agents and the real
  `UCODE_TEST_WORKSPACE` already used by e2e. The following rules cover every
  test, helper, and fixture in that directory.

## Integration rules: no test hacks

1. **No mocks or monkeypatching.** No `monkeypatch`, `pytest.MonkeyPatch`,
   `unittest.mock`, `Mock`, `MagicMock`, `patch`, replacement executables, fake
   HTTP services, import substitution, or runtime alteration of application code.
2. **Do not import application internals.** Exercise installed `ug` / `ucode`
   and agent commands through subprocesses. Do not call config writers,
   launchers, parsers, authentication helpers, or state helpers directly.
3. **Create ug state through public CLI commands.** Never hand-write ug state,
   cached discovery, or generated gateway config to get past setup. Ordinary
   input files and pre-existing user-owned settings are valid scenario inputs;
   identify them clearly and assert their preservation.
4. **No production changes just to make tests pass.** No test-only environment
   switches, special server branches, disabled validation, privileged-path
   overrides, or hardcoded success. Real bugs require normal production fixes
   and regression coverage. Report failures instead of concealing them.
5. **Real responses and binaries.** Pin requested ug and agent versions. Never
   substitute a missing binary/service. Reuse explicit e2e workspace/auth settings;
   never pick a developer's Databricks profile automatically.
6. **Fail honestly.** Missing prerequisites/capabilities, timeouts, protocol errors,
   and unexpected nonzero exits fail. Do not add skips, xfails, broad exception
   suppression, task retries, or weaker assertions to make CI green. Report
   deliberately selected subsets explicitly.
7. **Assert observable behavior.** Use actual files, structured final agent
   results, exit status and protocol exchanges. Process startup, banners, echoed
   input, or nonempty output alone do not establish a successful task.
8. **Isolate through processes and environments.** Fresh homes, working directories,
   package environments and explicit subprocess `env` are allowed. Do not mutate
   pytest's environment to redirect imported modules. Avoid developer config,
   credentials, OS-managed paths and installed tools.
9. **Bound and clean up work.** Give every process a timeout and explicit stdin;
   reap children on failure. Do not swallow timeouts or leave servers running.
   Keep live prompts small. Record models used by explicit-model scenarios;
   normal boot should use the workspace's own configuration.
10. **Evidence without secrets.** Record versions, dependencies, command shape,
    exit codes and redacted diagnostics. Never archive tokens, credential files,
    complete environments or developer homes.
11. **Drive the real TUI.** PTYs and terminal screen parsers are allowed. Handle
    onboarding and trust through visible UI choices and actual keystrokes; never
    seed onboarding completion, intercept model traffic, or replace a TUI with
    print/exec mode while claiming interactive coverage. Require an interactive
    prompt and observable input/exit behavior. Boot is not first-prompt inference.

`configure --skip-validate` in setup avoids an extra generic model prompt; task
tests then require a real independently asserted task. Configuration alone must
never be presented as inference coverage. Do not add shortcuts around the
behavior a test claims to exercise.

## Add / modify / remove

- **Add:** state the user scenario and affected versions, choose the category,
  add a focused test and coverage row. Demonstrate regression failure on the
  affected combination when it is available.
- **Modify:** preserve or strengthen assertions. Explain a changed product
  contract rather than silently redefining success. Update both coverage READMEs
  and scenario/version inputs when their claims change.
- **Remove:** explain obsolete/duplicate coverage and where any still-required
  behavior is tested. Mark remaining gaps as not covered. Never remove a case
  merely because a real agent or gateway currently fails it.
- Help is not desktop startup; configuration is not a completed task; routing
  bypass is not successful routing; launcher-style arguments are not Isaac.

## Verification

```bash
uv run pytest tests/test_integration_contract.py
uv run ruff check tests/ scripts/run_integration.py
uv run ruff format --check tests/ scripts/run_integration.py
python3 scripts/run_integration.py --help
```

Run integration through `scripts/run_integration.py` with explicit versions.
Use `--installation-only` for package checks and `-- -k EXPRESSION` for a reported
subset. Live checks require the e2e workspace and bearer/profile. Model overrides
are optional inputs for reproducing an explicit-model failure.
Report passes, failures and what was not run. Collection/lint/package checks
are not evidence of a live integration pass.
