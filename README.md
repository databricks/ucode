# Databricks Coding Gateway

`coding-gateway` is a lightweight launcher for running Codex, Claude Code, and Gemini CLI through Databricks.

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

This writes managed config files for all three tools (`~/.codex/config.toml`, `~/.claude/settings.json`, `~/.gemini/.env`).

### 2. Configure MCP servers (optional)

```bash
coding-gateway configure mcp
```

Add Databricks MCP servers to installed MCP-capable tools: Codex, Claude Code, Gemini CLI, and OpenCode.
Options are shown in this order:

- Discovered external MCP connections
- Databricks SQL
- Built-in AI tools
- Other Databricks MCP products
- Custom MCP server URL

Discovered external MCP connections are listed directly. MCP auth uses a Databricks token that
`coding-gateway` sets when launching each tool.

### 3. Launch an agent

```bash
coding-gateway                    # launches Codex (default)
coding-gateway --agent claude
coding-gateway --agent gemini
```

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

Existing files are backed up before being overwritten. `coding-gateway revert` restores backups.

## Authentication

- Databricks authentication uses OAuth via `databricks auth login`
- Codex and Claude use a Databricks token helper (no fixed token stored)
- Gemini refreshes its bearer token automatically while running through `coding-gateway`

## Documentation

- [Databricks AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-beta)
- [Databricks AI Gateway coding agent integration](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-beta)
- [Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication)
- [Monitor AI Gateway usage](https://docs.databricks.com/aws/en/ai-gateway/configure-ai-gateway-endpoints#track-usage-of-an-endpoint)

## Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request against `main`.

## Security

Please report security vulnerabilities to security@databricks.com rather than opening a public issue.

## License

See [LICENSE.md](./LICENSE.md) and [NOTICE.md](./NOTICE.md).
