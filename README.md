# Unity AI Gateway Coding CLI (ucode)

`ucode` is a lightweight launcher for running Codex, Claude Code, Gemini CLI, OpenCode, GitHub Copilot CLI, and Pi through Databricks.

## Requirements

- Python 3.12+ — install with `uv` ([uv.astral.sh](https://docs.astral.sh/uv/getting-started/installation/))
- `npm` if tool CLIs need to be installed automatically

## Installation

```bash
uv tool install git+https://github.com/databricks/ucode
```

---

## Usage

Just run the tool you want:

```bash
ucode codex      # OpenAI Codex
ucode claude     # Claude Code
ucode gemini     # Gemini CLI
ucode opencode   # OpenCode
ucode copilot    # GitHub Copilot CLI
ucode pi         # Pi
```

On first launch, `ucode` will prompt for your Databricks workspace URL, authenticate, and configure that tool automatically. Subsequent launches go straight to the agent.

Pass flags directly to the underlying tool:

```bash
ucode claude -r          # resume last session
ucode codex --full-auto
```

All agents route through Databricks AI Gateway using your workspace credentials — no API keys required.

To configure all tools at once:

```bash
ucode configure
```

To configure specific tools without the picker, pass a comma-separated list:

```bash
ucode configure --agents claude,codex
```

Available agent names are `codex`, `claude`, `gemini`, `opencode`, `copilot`, and `pi`.

### MCP servers (optional)

```bash
ucode configure mcp
```

Add Databricks MCP servers to installed MCP-capable tools: Codex, Claude Code, Gemini CLI, OpenCode, and GitHub Copilot CLI.
Options are shown in this order:

- Discovered external MCP connections
- Databricks SQL
- Managed Databricks MCPs (Vector Search, UC Functions, etc.)
- Custom MCP server URL

Discovered external MCP connections are listed directly. MCP auth uses a Databricks token that
`ucode` sets when launching each tool.

---

## Other Commands

| Command | Description |
|---------|-------------|
| `ucode status` | Show current workspace, base URLs, managed config files, and selected models |
| `ucode usage` | Show AI Gateway usage summary |
| `ucode revert` | Clear saved state and restore backed-up config files |
| `ucode configure --dry-run` | Preview config files without writing them |
| `ucode configure --agents claude,codex` | Configure specific agents without the interactive picker |

## Managed Local Files

`ucode` manages these files:

| File | Tool |
|------|------|
| `~/.codex/config.toml` | Codex |
| `~/.claude/settings.json` | Claude Code |
| `~/.gemini/.env` | Gemini CLI |
| `~/.config/opencode/opencode.json` | OpenCode |
| `~/.copilot/.env` | GitHub Copilot CLI |
| `~/.pi/agent/models.json` | Pi |

Existing files are backed up before being overwritten. `ucode revert` restores backups.


## Documentation

- [Databricks AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-beta)
- [Databricks AI Gateway coding agent integration](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-beta)
- [Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication)
- [Monitor AI Gateway usage](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints#track-usage-of-an-endpoint)

## Contributing

Contributions are welcome.

### Getting started

```bash
git clone https://github.com/databricks/ucode
cd ucode
uv sync
```

### Development workflow

1. Create a feature branch off `main`.
2. Make your changes — keep them scoped to the requested behavior.
3. Run the test suite before pushing:

   ```bash
   uv run pytest          # unit tests
   uv run ruff check .    # lint
   ```

4. For end-to-end testing against a real workspace:

   ```bash
   UCODE_TEST_WORKSPACE=<db_workspace_url> uv run pytest tests/test_e2e.py -v
   ```

5. Open a pull request against `main`.

### Adding a new agent

- Add `src/ucode/agents/<name>.py` with at least `write_tool_config`, `launch`, `default_model`, and `validate_cmd`.
- Register it in `src/ucode/agents/__init__.py`.
- Add focused tests under `tests/`.

## Security

Please report security vulnerabilities to security@databricks.com rather than opening a public issue.

## License

See [LICENSE.md](./LICENSE.md) and [NOTICE.md](./NOTICE.md).
