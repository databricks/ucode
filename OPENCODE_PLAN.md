# Plan: Add opencode to ucode

## What is opencode

opencode is an AI coding CLI (`npm i -g opencode-ai`) that uses the Vercel AI SDK
internally. Unlike codex/claude/gemini which each speak one proprietary protocol,
opencode supports 22+ AI SDK providers via an `npm` field in its JSON config.

Config lives at `~/.config/opencode/opencode.json`.

---

## Why it's not as simple as the other tools

### 1. Token auth — no shell command support (TBD)

Codex and Claude Code support a shell auth command that runs on every request:
- Codex: `auth.command = "sh"` in TOML
- Claude Code: `apiKeyHelper` shell script in settings.json

This means their tokens are always fresh. Gemini doesn't support this, so
ucode spawns a background thread to refresh `GEMINI_API_KEY` in the
.env file every 30 minutes.

**opencode config only supports static apiKey or `{env:VAR}` syntax.**
This means we need to either:
- (a) Embed a live token at launch time and refresh like Gemini — requires
      background refresh thread and subprocess launch (not execvp)
- (b) Investigate whether opencode re-reads its config on each request (unlikely)
- (c) Set `DATABRICKS_TOKEN` env var and use `{env:DATABRICKS_TOKEN}` in config —
      then refresh that env var in the background (cleanest if it works)

**Open question**: Does opencode read apiKey from env at request time or only at
startup? If at request time, option (c) works cleanly.

### 2. Multiple providers vs. single endpoint

opencode can be configured with multiple providers. Two approaches:

**Option A — Single mlflow endpoint with `@ai-sdk/openai-compatible`**
```json
{
  "provider": {
    "databricks": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "https://<workspace>/ai-gateway/mlflow/v1",
        "apiKey": "{env:DATABRICKS_TOKEN}"
      }
    }
  },
  "model": "databricks/databricks-claude-sonnet-4-6"
}
```
- Simple, one endpoint to check
- Some models have known incompatibilities (content-as-array for Gemini 3.x thinking)
- User picks from `databricks-*` model IDs

**Option B — Dedicated provider endpoints per model family**
```json
{
  "provider": {
    "databricks-anthropic": {
      "npm": "@ai-sdk/anthropic",
      "options": {
        "baseURL": "https://<workspace>/ai-gateway/anthropic",
        "apiKey": "{env:DATABRICKS_TOKEN}"
      }
    },
    "databricks-google": {
      "npm": "@ai-sdk/google",
      "options": {
        "baseURL": "https://<workspace>/ai-gateway/gemini",
        "apiKey": "{env:DATABRICKS_TOKEN}"
      }
    },
    "databricks-openai": {
      "npm": "@ai-sdk/openai",
      "options": {
        "baseURL": "https://<workspace>/ai-gateway/codex/v1",
        "apiKey": "{env:DATABRICKS_TOKEN}"
      }
    }
  },
  "model": "databricks-anthropic/databricks-claude-sonnet-4-6"
}
```
- Native SDK per family → better compatibility (thinking models work correctly)
- More config complexity — 3 providers, 3 endpoints to check
- User picks a model + its provider is inferred from the model name prefix

**Open question**: Does `@ai-sdk/anthropic` work against the Databricks Anthropic
gateway with `databricks-*` model IDs? Same for `@ai-sdk/google` against the
Gemini gateway? Needs testing.

### 3. Model selection

Unlike codex (no model selection), opencode needs a model specified in config.
The model list should come from the `providers/databricks/models/` TOMLs in
models.dev (or from querying the workspace endpoint directly).

`classify_tool_from_text()` and `discover_workspace_models()` would need an
"opencode" case, but since opencode supports ANY model family, the full
`databricks-*` list is valid — not just one family like claude/gemini.

Alternatively: skip workspace discovery entirely, let user type a model name,
show the list from models.dev as a hint.

### 4. Config path

`~/.config/opencode/opencode.json` — different from the other tools which use
home-dir dotfiles. Needs new path constants.

### 5. Usage tracking

The Spark SQL query in `build_usage_report_query()` filters by user_agent
containing "codex", "claude", or "gemini". Need to verify opencode sends
"opencode" in its user-agent (likely yes, confirmed from worker.ts in models.dev).
Add "opencode" case to the query.

### 6. Gateway endpoint check

`check_gateway_endpoint()` needs an opencode case. For Option A (mlflow), probe
`/ai-gateway/mlflow/v1/models`. For Option B, probe each provider endpoint.

### 7. Validation test command

`validate_tool()` needs an opencode case. opencode's CLI args are TBD — need to
check if it supports a single `-p "prompt"` style invocation or requires
interactive mode only.

---

## What changes in cli.py

| Section | Change |
|---|---|
| Path constants | Add `OPENCODE_CONFIG_DIR`, `OPENCODE_CONFIG_PATH`, `OPENCODE_BACKUP_PATH` |
| `TOOL_SPECS` | Add opencode entry: binary, package, display, config_path, backup_path |
| `TOOL_ALIASES` | Add "opencode" alias |
| `DEFAULT_SELECTED_MODELS` | Add default model (e.g. `databricks-claude-sonnet-4-6`) |
| `build_tool_base_url()` | Add opencode case (mlflow endpoint or per-family) |
| `render_opencode_config()` | New function — generates opencode.json content |
| `write_opencode_tool_config()` | New function — backup/write/mark managed |
| `configure_tool()` | Add opencode elif branch |
| `check_gateway_endpoint()` | Add opencode elif branch |
| `validate_tool()` | Add opencode elif branch |
| `build_usage_report_query()` | Add opencode to user_agent filters |
| `classify_tool_from_text()` | Add "opencode" detection |
| Launch path | Either reuse `launch_tool()` (execvp) or new `launch_opencode_tool()` with token refresh |

---

## Open questions to resolve before implementing

1. **Does opencode re-read `{env:VAR}` at request time or only at startup?**
   → Determines auth approach (static refresh vs. env var)

2. **Does `@ai-sdk/anthropic` work against `/ai-gateway/anthropic` with `databricks-*` model IDs?**
   → Determines Option A vs. Option B for provider config

3. **What are opencode's CLI flags for non-interactive single-prompt invocation?**
   → Needed for `validate_tool()` test command

4. **Does opencode emit "opencode" in its user-agent?**
   → Needed for usage tracking SQL

5. **Which approach for model selection?**
   → Workspace discovery (requires opencode case in classify_tool_from_text)
     or hardcoded list from models.dev

---

## Recommended next steps

1. Test Option B (native provider SDKs) against Databricks endpoints — run a quick
   script using `@ai-sdk/anthropic` with baseURL set to the Databricks Anthropic
   gateway to confirm model IDs and auth work
2. Check opencode source for env var re-read behavior
3. Check opencode CLI flags for non-interactive mode
4. Decide Option A vs B, then implement the 12 changes above
