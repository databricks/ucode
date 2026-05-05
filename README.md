# Databricks Coding Gateway

`coding-gateway` is a lightweight launcher for running Codex, Claude Code, Gemini CLI, and OpenCode through Databricks.

## Requirements

- Python 3.12+ — install with `uv` ([uv.astral.sh](https://docs.astral.sh/uv/getting-started/installation/))
- `npm` if tool CLIs need to be installed automatically

## Installation

```bash
uv tool install git+https://github.com/databricks/coding-gateway
```

---

## Usage

Just run the tool you want:

```bash
coding-gateway codex      # OpenAI Codex
coding-gateway claude     # Claude Code
coding-gateway gemini     # Gemini CLI
coding-gateway opencode   # OpenCode
```

On first launch, `coding-gateway` will prompt for your Databricks workspace URL, authenticate, and configure that tool automatically. Subsequent launches go straight to the agent.

Pass flags directly to the underlying tool:

```bash
coding-gateway claude -r          # resume last session
coding-gateway codex --full-auto
```

All agents route through Databricks AI Gateway using your workspace credentials — no API keys required.

To configure all tools at once:

```bash
coding-gateway configure
```

### MCP servers (optional)

```bash
coding-gateway configure mcp
```

Add Databricks MCP servers to Claude Code. Supported server types:

- **External** — e.g. confluence-mcp, jira-mcp
- **UC Functions** — Unity Catalog AI functions
- **Genie** — AI/BI dashboards
- **Custom** — any MCP server URL

You will be prompted for OAuth credentials (client ID and secret) that are reused for all servers added in the session.

---

## Other Commands

| Command | Description |
|---------|-------------|
| `coding-gateway status` | Show current workspace, base URLs, managed config files, and selected models |
| `coding-gateway usage` | Show AI Gateway usage summary |
| `coding-gateway revert` | Clear saved state and restore backed-up config files |
| `coding-gateway configure --dry-run` | Preview config files without writing them |

## Managed Local Files

`coding-gateway` manages these files:

| File | Tool |
|------|------|
| `~/.codex/config.toml` | Codex |
| `~/.claude/settings.json` | Claude Code |
| `~/.gemini/.env` | Gemini CLI |
| `~/.config/opencode/opencode.json` | OpenCode |

Existing files are backed up before being overwritten. `coding-gateway revert` restores backups.


## Documentation

- [Databricks AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-beta)
- [Databricks AI Gateway coding agent integration](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-beta)
- [Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication)
- [Monitor AI Gateway usage](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints#track-usage-of-an-endpoint)

## Contributing

Contributions are welcome.

### Getting started

```bash
git clone https://github.com/databricks/coding-gateway
cd coding-gateway
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
   CODING_GATEWAY_TEST_WORKSPACE=<db_workspace_url> uv run pytest tests/test_e2e.py -v
   ```

5. Open a pull request against `main`.

### Adding a new agent

- Add `src/coding_tool_gateway/agents/<name>.py` with at least `write_tool_config`, `launch`, `default_model`, and `validate_cmd`.
- Register it in `src/coding_tool_gateway/agents/__init__.py`.
- Add focused tests under `tests/`.

## Security

Please report security vulnerabilities to security@databricks.com rather than opening a public issue.

## License

See [LICENSE.md](./LICENSE.md) and [NOTICE.md](./NOTICE.md).
