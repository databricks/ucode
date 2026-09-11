"""Install a reproducible ug/agent combination and run black-box integration tests.

Uses fresh virtualenvs and an isolated npm prefix, never the checkout's uv.lock
or the developer's installed agents. Only the live workspace is shared with e2e.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_PACKAGES = {"claude": "@anthropic-ai/claude-code", "codex": "@openai/codex"}


@contextlib.contextmanager
def managed_process(command, *, interrupt=False, **kwargs):
    """Bound child lifetimes, including descendants that outlive their parent."""
    proc = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        yield proc
    finally:
        # Give pytest a KeyboardInterrupt so its fixtures can clean up the
        # separate process groups used by agent commands before pytest exits.
        first_signal = signal.SIGINT if interrupt else signal.SIGTERM
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, first_signal)
        try:
            proc.wait(timeout=15 if interrupt else 5)
        except subprocess.TimeoutExpired:
            pass
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def exact_npm_version(value: str) -> str:
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", value):
        raise argparse.ArgumentTypeError(
            "Use an exact version, for example 2.1.268; not latest/^/~."
        )
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--ug-version", default="checkout", help="Exact ucode release, or checkout."
    )
    source.add_argument(
        "--ug-wheel", type=Path, help="Previously built wheel to reproduce a release."
    )
    parser.add_argument("--entry-point", choices=["ug", "ucode"], default="ug")
    parser.add_argument("--claude-version", type=exact_npm_version)
    parser.add_argument("--codex-version", type=exact_npm_version)
    parser.add_argument("--claude-model", default=os.environ.get("UG_INTEGRATION_CLAUDE_MODEL"))
    parser.add_argument("--codex-model", default=os.environ.get("UG_INTEGRATION_CODEX_MODEL"))
    parser.add_argument("--python", default=sys.executable, help="Python 3.12+ path or uv version.")
    parser.add_argument("--dependency", action="append", default=[], metavar="PACKAGE==VERSION")
    parser.add_argument("--constraints", type=Path, help="Replay a previous dependencies.txt.")
    parser.add_argument(
        "--npm-lock", type=Path, help="Replay a previous npm-lock.json with npm ci."
    )
    parser.add_argument(
        "--index-url", default=os.environ.get("UV_INDEX_URL", "https://pypi.org/simple")
    )
    parser.add_argument("--npm-registry", default="https://registry.npmjs.org")
    parser.add_argument("--profile", help="Explicit Databricks profile to mint the live bearer.")
    parser.add_argument("--workspace", default=os.environ.get("UCODE_TEST_WORKSPACE"))
    parser.add_argument("--output", type=Path, help="New results directory; never reused.")
    parser.add_argument("--installation-only", action="store_true", help="No workspace calls.")
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="After --, pass pytest filters."
    )
    args = parser.parse_args()
    # Only selection/early-stop controls are accepted. Pytest configuration,
    # plugins and report destinations are part of the suite's isolation contract.
    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("-k")
    filters.add_argument("-m")
    filters.add_argument("-x", action="store_true")
    filters.add_argument("--maxfail", type=int)
    extra = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    selected = filters.parse_args(extra)
    marker = selected.m
    if args.installation_only:
        marker = f"installation and ({marker})" if marker else "installation"
    args.pytest_args = []
    for flag, value in (("-k", selected.k), ("-m", marker), ("--maxfail", selected.maxfail)):
        if value is not None:
            args.pytest_args.extend([flag, str(value)])
    if selected.x:
        args.pytest_args.append("-x")
    if not (args.claude_version or args.codex_version):
        parser.error("Select --claude-version and/or --codex-version explicitly.")
    if args.ug_version != "checkout" and not re.fullmatch(
        r"[0-9][0-9A-Za-z.!+_-]*", args.ug_version
    ):
        parser.error(
            "--ug-version must be an exact release, or checkout; use --ug-wheel for a file."
        )
    for dependency in args.dependency:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+-]+", dependency):
            parser.error("--dependency requires an exact PACKAGE==VERSION constraint.")
    if not args.installation_only:
        if not args.workspace or not args.workspace.startswith("https://"):
            parser.error(
                "Set UCODE_TEST_WORKSPACE to the existing e2e workspace, or use --workspace."
            )
        if not (args.profile or os.environ.get("DATABRICKS_BEARER", "").strip()):
            parser.error("Provide the e2e DATABRICKS_BEARER or select --profile explicitly.")
    return args


def main() -> int:
    args = arguments()
    if os.name != "posix":
        raise SystemExit("This runner supports Linux and macOS. Use the container on other hosts.")

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    binaries = {name: shutil.which(name) for name in ("uv", "npm", "node", "databricks")}
    required = ["uv", "npm", "node"] + ([] if args.installation_only else ["databricks"])
    missing = [name for name in required if not binaries[name]]
    if missing:
        raise SystemExit("Install these prerequisites first: " + ", ".join(missing))
    if not args.installation_only:
        managed_paths = []
        if args.codex_version:
            managed_paths.extend(
                [Path("/etc/codex/managed_config.toml"), Path("/etc/codex/requirements.toml")]
            )
        if args.claude_version:
            managed_paths.append(
                Path(
                    "/Library/Application Support/ClaudeCode/managed-settings.json"
                    if sys.platform == "darwin"
                    else "/etc/claude-code/managed-settings.json"
                )
            )
        present = [str(path) for path in managed_paths if path.exists()]
        if present:
            raise SystemExit(
                "Machine-wide agent settings can override the selected test workspace. "
                "Use the integration container instead of this host: " + ", ".join(present)
            )

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / ".integration-runs" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    build_log = output / "install.log"
    # Do not inherit project environments, resolver constraints, pytest options,
    # Python optimization, agent credentials, or npm settings from the caller.
    keep = (
        "PATH",
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
    base_env = {key: os.environ[key] for key in keep if key in os.environ}
    build_home = output / "build-home"
    build_home.mkdir()
    base_env["HOME"] = str(build_home)
    base_env["USERPROFILE"] = str(build_home)
    base_env["npm_config_cache"] = str(output / "npm-cache")
    base_env["npm_config_fetch_retries"] = "1"
    base_env["npm_config_fetch_timeout"] = "30000"
    base_env["UV_CACHE_DIR"] = str(output / "cache")
    base_env["UV_INDEX_URL"] = args.index_url
    bearer = os.environ.get("DATABRICKS_BEARER", "").strip()

    def redact(value: str) -> str:
        return value.replace(bearer, "<redacted>") if bearer else value

    def run(command, *, cwd=output, env=base_env, timeout=600) -> str:
        timed_out = False
        with managed_process(
            [str(x) for x in command],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as proc:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
        if timed_out:
            stdout, stderr = proc.communicate(timeout=5)
        with build_log.open("a") as log:
            log.write(redact(stdout + stderr))
        if timed_out:
            raise RuntimeError(f"{command[0]} exceeded {timeout}s; see {build_log}")
        if proc.returncode:
            raise RuntimeError(f"{command[0]} failed; see {build_log}\n" + redact(stderr[-2000:]))
        return stdout.strip()

    report = {
        "requested": {
            "ug": str(args.ug_wheel) if args.ug_wheel else args.ug_version,
            "entry_point": args.entry_point,
            "claude": args.claude_version,
            "codex": args.codex_version,
            "claude_model": args.claude_model,
            "codex_model": args.codex_model,
            "dependencies": args.dependency,
            "workspace": args.workspace,
        },
        "platform": platform.platform(),
        "installation_only": args.installation_only,
    }
    manifest = output / "versions.json"
    exitcode = 1
    try:
        print(f"Installing selected versions. Results: {output}", flush=True)
        uv = binaries["uv"]
        runtime, testenv = output / "ug-runtime", output / "test-runtime"
        for path in (runtime, testenv):
            run([uv, "venv", "--python", args.python, path])
        python = runtime / "bin/python"
        report["python"] = run([python, "--version"])
        report["uv"] = run([uv, "--version"])
        report["node"] = run([binaries["node"], "--version"])
        report["npm"] = run([binaries["npm"], "--version"])
        if binaries["databricks"]:
            report["databricks"] = run([binaries["databricks"], "--version"])

        if args.ug_wheel:
            wheel = args.ug_wheel.resolve()
            if not wheel.is_file() or wheel.suffix != ".whl":
                raise RuntimeError(f"Wheel does not exist: {wheel}")
        elif args.ug_version == "checkout":
            if not (ROOT / "pyproject.toml").is_file():
                raise RuntimeError(
                    "No checkout in this image. Pass --ug-version or mount --ug-wheel."
                )
            wheels = output / "wheels"
            run([uv, "build", "--wheel", "--out-dir", wheels, ROOT], cwd=ROOT)
            (wheel,) = wheels.glob("*.whl")
            report["git_commit"] = run(["git", "rev-parse", "HEAD"], cwd=ROOT)
            report["tracked_diff"] = run(
                ["git", "diff", "HEAD", "--", "src", "pyproject.toml"], cwd=ROOT
            )
        else:
            wheel = None

        if wheel:
            report["wheel_sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        package = str(wheel) if wheel else f"ucode=={args.ug_version}"
        constraints = output / "requested-constraints.txt"
        constraints.write_text(
            (args.constraints.read_text() if args.constraints else "")
            + "\n"
            + "\n".join(args.dependency)
        )
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--index-url",
                args.index_url,
                "--constraint",
                constraints,
                package,
            ]
        )
        run([uv, "pip", "check", "--python", python])
        freeze = run([uv, "pip", "freeze", "--python", python])
        (output / "installed.txt").write_text(freeze + "\n")
        (output / "dependencies.txt").write_text(
            "\n".join(
                line for line in freeze.splitlines() if not re.match(r"ucode(?:==|\s*@)", line)
            )
            + "\n"
        )
        # The report proves we imported site-packages, not src/ via an editable install.
        report["package"] = json.loads(
            run(
                [
                    python,
                    "-c",
                    (
                        "import importlib.metadata as m, json, ucode; "
                        "print(json.dumps({'version': m.version('ucode'), 'path': ucode.__file__}))"
                    ),
                ]
            )
        )
        package_path = Path(report["package"]["path"]).resolve()
        if not package_path.is_relative_to(runtime):
            raise RuntimeError(
                f"Application was imported outside its isolated environment: {package_path}"
            )
        binary = runtime / "bin" / args.entry_point
        if not binary.is_file():
            raise RuntimeError(
                f"Selected release has no {args.entry_point} entry point; try --entry-point ucode."
            )

        agents = [agent for agent in AGENT_PACKAGES if getattr(args, f"{agent}_version")]
        npm_prefix = output / "agents"
        npm_prefix.mkdir()
        if args.npm_lock:
            (npm_prefix / "package.json").write_text(
                json.dumps(
                    {
                        "dependencies": {
                            AGENT_PACKAGES[a]: getattr(args, f"{a}_version") for a in agents
                        }
                    }
                )
            )
            shutil.copyfile(args.npm_lock, npm_prefix / "package-lock.json")
            run(
                [
                    binaries["npm"],
                    "ci",
                    "--prefix",
                    npm_prefix,
                    "--no-audit",
                    "--no-fund",
                    "--registry",
                    args.npm_registry,
                ]
            )
        else:
            run(
                [
                    binaries["npm"],
                    "install",
                    "--prefix",
                    npm_prefix,
                    "--no-audit",
                    "--no-fund",
                    "--save-exact",
                    "--registry",
                    args.npm_registry,
                    *[f"{AGENT_PACKAGES[a]}@{getattr(args, f'{a}_version')}" for a in agents],
                ]
            )
        shutil.copyfile(npm_prefix / "package-lock.json", output / "npm-lock.json")
        report["npm_packages"] = json.loads(
            run(
                [
                    binaries["npm"],
                    "ls",
                    "--prefix",
                    npm_prefix,
                    "--depth=0",
                    "--json",
                ]
            )
        )["dependencies"]
        agent_bin = npm_prefix / "node_modules/.bin"
        # Expose only selected executables to tested programs; other installed
        # developer agents cannot be discovered accidentally via inherited PATH.
        tool_bin = output / "tools"
        tool_bin.mkdir()
        for name in ("node", "databricks"):
            if binaries[name]:
                (tool_bin / name).symlink_to(binaries[name])
        runtime_env = dict(base_env)
        runtime_env["PATH"] = os.pathsep.join(
            map(str, [runtime / "bin", agent_bin, tool_bin, "/usr/bin", "/bin"])
        )
        report["agents"] = {}
        for agent in agents:
            version = run([agent_bin / agent, "--version"], env=runtime_env, timeout=30)
            expected = getattr(args, f"{agent}_version")
            if not re.search(rf"(?<![\w.]){re.escape(expected)}(?![\w.])", version):
                raise RuntimeError(f"Expected {agent} {expected}, got {version!r}")
            report["agents"][agent] = version

        if args.profile and not args.installation_only:
            # Never select a local profile implicitly. Do not persist auth output.
            with managed_process(
                [
                    binaries["databricks"],
                    "auth",
                    "token",
                    "--host",
                    args.workspace,
                    "--profile",
                    args.profile,
                    "--output",
                    "json",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                stdin=subprocess.DEVNULL,
            ) as auth:
                auth_stdout, _ = auth.communicate(timeout=30)
            if auth.returncode:
                raise RuntimeError(
                    "Could not obtain a token for the selected profile; log in first."
                )
            bearer = json.loads(auth_stdout).get("access_token", "")
            if not bearer:
                raise RuntimeError("Selected profile returned no access token.")

        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                testenv / "bin/python",
                "--index-url",
                args.index_url,
                "pytest==9.0.3",
                "pexpect==4.9.0",
                "pyte==0.8.2",
            ]
        )
        (output / "test-dependencies.txt").write_text(
            run([uv, "pip", "freeze", "--python", testenv / "bin/python"]) + "\n"
        )
        runtime_env.update(
            {
                "UG_INTEGRATION_BIN": str(binary),
                "UG_INTEGRATION_RUN_DIR": str(output),
                "UG_INTEGRATION_AGENTS": ",".join(agents),
                "UCODE_TEST_WORKSPACE": args.workspace or "",
                "DATABRICKS_BEARER": bearer,
            }
        )
        for agent in agents:
            runtime_env[f"UG_INTEGRATION_{agent.upper()}_MODEL"] = (
                getattr(args, f"{agent}_model") or ""
            )
        suite = ROOT / "tests/integration"
        suite_hash = hashlib.sha256()
        for path in [Path(__file__), *sorted(suite.glob("*.py")), suite / "pytest.ini"]:
            suite_hash.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
        report["suite_sha256"] = suite_hash.hexdigest()
        extra = args.pytest_args
        report["pytest_args"] = extra
        manifest.write_text(redact(json.dumps(report, indent=2)) + "\n")
        print("Running integration tests against the installed package.", flush=True)
        with managed_process(
            [
                testenv / "bin/python",
                "-m",
                "pytest",
                "-c",
                suite / "pytest.ini",
                f"--confcutdir={suite}",
                suite,
                "-v",
                "-o",
                f"cache_dir={output / 'pytest-cache'}",
                f"--junitxml={output / 'junit.xml'}",
                *extra,
            ],
            env=runtime_env,
            cwd=output,
            stdin=subprocess.DEVNULL,
            interrupt=True,
        ) as result:
            result.wait(timeout=3600)
        exitcode = result.returncode
        junit = output / "junit.xml"
        if junit.is_file():
            suites = ET.parse(junit).getroot().iter("testsuite")
            totals = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
            for suite_result in suites:
                for key in totals:
                    totals[key] += int(suite_result.get(key, "0"))
            report["results"] = totals
            if totals["skipped"]:
                raise RuntimeError("Requested integration tests were skipped; see junit.xml.")
            if not totals["tests"] and not exitcode:
                raise RuntimeError(
                    "No integration tests executed; see the selected pytest filters."
                )
        elif not exitcode:
            raise RuntimeError("Pytest returned success without a test report.")
        # A bootstrap/update path must not silently alter the selected agent version.
        for agent in agents:
            after = run([agent_bin / agent, "--version"], env=runtime_env, timeout=30)
            if after != report["agents"][agent]:
                raise RuntimeError(f"{agent} changed version during the suite: {after}")
    except KeyboardInterrupt:
        report["error"] = "Integration run interrupted."
        exitcode = 130
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        report["error"] = redact(str(exc))
        print(report["error"], file=sys.stderr)
        exitcode = 1
    finally:
        report["exitcode"] = exitcode
        manifest.write_text(redact(json.dumps(report, indent=2)) + "\n")
        print(f"Integration results: {output}", flush=True)
    return exitcode


if __name__ == "__main__":
    raise SystemExit(main())
