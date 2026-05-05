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

## Setup

### 1. Configure the workspace

```bash
coding-gateway configure
```

Enter your Databricks workspace URL. `coding-gateway` automatically detects whether Databricks AI Gateway is available and configures tool endpoints accordingly.

This writes managed config files for each tool (`~/.codex/config.toml`, `~/.claude/settings.json`, `~/.gemini/.env`, `~/.config/opencode/opencode.json`).

### 2. Launch an agent

```bash
coding-gateway codex
coding-gateway claude
coding-gateway gemini
coding-gateway opencode
```

If a tool hasn't been configured yet, running it will automatically configure it for you.

### 3. Configure MCP servers (optional)

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

## Usage

Once configured, launch any supported agent directly from your terminal:

```bash
coding-gateway codex
coding-gateway claude
coding-gateway gemini
coding-gateway opencode
```

Pass flags directly to the underlying tool:

```bash
coding-gateway claude -r          # resume last session
coding-gateway codex --full-auto
```

All agents route through Databricks AI Gateway using your workspace credentials — no API keys required.

---

## Other Commands

| Command | Description |
|---------|-------------|
| `coding-gateway status` | Show current workspace, base URLs, managed config files, and selected models |
| `coding-gateway usage` | Show AI Gateway usage summary |
| `coding-gateway revert` | Clear saved state and restore backed-up config files |
| `coding-gateway configure --dry-run` | Preview config files without writing them |

## Usage Reporting

```bash
coding-gateway usage
```

Requires Databricks AI Gateway. Queries `system.ai_gateway.usage` and shows:

- Token totals for today, last 7 days, and last 30 days
- Active tools and top models this week
- 7-day breakdown per tool (Codex, Claude Code, Gemini CLI)

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
