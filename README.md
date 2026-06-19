# voxcode — Van Oord OpenCode Launcher

`voxcode` is a thin launcher for running [OpenCode](https://opencode.ai) through the Databricks AI Gateway. It routes all LLM traffic through your Databricks workspace — no API keys required.

**Only models approved by the platform team are available.** The allowlist is maintained in `src/voxcode/allowed_models.py`.

## Requirements

- Python 3.12+ — install with `uv` ([uv.astral.sh](https://docs.astral.sh/uv/getting-started/installation/))
- `npm` (for OpenCode CLI auto-install)

## Installation

```bash
uv tool install git+https://github.com/vanoord/voxcode
```

---

## Usage

```bash
voxcode opencode     # Launch OpenCode
voxcode launch       # Same as above (alias)
```

On first launch, `voxcode` will prompt for your Databricks workspace URL, authenticate, and configure OpenCode automatically. Subsequent launches go straight to the agent.

Pass flags directly to OpenCode:

```bash
voxcode opencode --model databricks-anthropic/claude-sonnet-4-20250514
```

All traffic routes through Databricks AI Gateway using your workspace credentials.

To configure:

```bash
voxcode configure
```

To configure without the workspace picker:

```bash
voxcode configure --workspaces https://your-workspace.azuredatabricks.net
```

Alternatively, use existing Databricks CLI profiles:

```bash
voxcode configure --profiles DEFAULT
```

For CI or headless environments (PAT-based auth):

```bash
voxcode configure --profiles DEFAULT --use-pat --skip-validate --skip-upgrade
```

### MCP servers (optional)

```bash
voxcode configure mcp
```

Add Databricks MCP servers to OpenCode (SQL, Vector Search, UC Functions, Genie spaces, etc.).

---

## Other Commands

| Command | Description |
|---------|-------------|
| `voxcode status` | Show current workspace, config, and selected models |
| `voxcode usage` | Show AI Gateway usage summary |
| `voxcode revert` | Clear saved state and restore backed-up config files |
| `voxcode configure --dry-run` | Preview config files without writing them |
| `voxcode configure mcp` | Add Databricks MCP servers |
| `voxcode configure tracing` | Enable MLflow tracing |
| `voxcode upgrade` | Upgrade voxcode to latest version |

## Managed Local Files

| File | Purpose |
|------|------|
| `~/.voxcode/state.json` | Workspace state |
| `~/.voxcode/opencode-xdg/opencode/opencode.json` | OpenCode config |

Existing files are backed up before being overwritten. `voxcode revert` restores backups.

## Approved Models

The platform team maintains the model allowlist in `src/voxcode/allowed_models.py`. Only models in this list are discoverable and launchable. Currently approved:

**Anthropic (via `databricks-anthropic` provider):**
- `claude-sonnet-4-20250514`
- `claude-haiku-4-20250514`

**Google (via `databricks-google` provider):**
- `gemini-2.5-pro`
- `gemini-2.5-flash`

To request a model addition, contact the platform team.

## Documentation

- [Databricks AI Gateway overview](https://docs.databricks.com/aws/en/ai-gateway/overview-beta)
- [Databricks AI Gateway coding agent integration](https://docs.databricks.com/aws/en/ai-gateway/coding-agent-integration-beta)
- [Databricks CLI authentication](https://docs.databricks.com/aws/en/dev-tools/cli/authentication)

## Development

```bash
git clone https://github.com/vanoord/voxcode
cd voxcode
uv sync
```

Run tests:

```bash
uv run pytest
uv run ruff check .
```

## License

See [LICENSE.md](./LICENSE.md) and [NOTICE.md](./NOTICE.md).
