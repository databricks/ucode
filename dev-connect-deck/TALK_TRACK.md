# Talk Track — Governing Coding Agent Sprawl (Dev Connect)

**Length:** ~20 min · **Speaker:** Rohit Agrawal (solo) · **Slides:** 18

**Pacing:** ~17–18 min total. Setup ~5 min → Solution ~2 min → Demo ~5 min (dev ~3, admin ~2) → Proof + close ~5 min. Keep setup brisk (~40 sec/slide) so the demo and proof land.
This is the developer-leaning, fast version. The IT-nightmare, "how do we bridge,"
and "what everyone wants" slides are cut — their points are folded into the bullets
below so the story still lands.

**Through-line:** The frontier moves every week, so the winning move is using *many*
agents — but every agent is its own silo (a dev problem) and its own privileged user
(a risk). **Omnigent** on **Unity AI Gateway** gives developers freedom and admins
visibility, and Databricks runs on it at 2,200-engineer scale.

---

## Setup (~5 min)

### 1. Title — "Governing Coding Agent Sprawl"
- One-line intro: SWE on Databricks AI; I build this platform and live in it every day.
- The tension in a sentence: developers want every new agent; orgs need control — this talk is how to get both.
- Promise a live demo of the developer *and* admin experience, plus how Databricks runs it.
- Keep it tight — you've got 20 minutes and the demo is the star.

### 2. A new coding agent every week
- The frontier moves weekly — new agents and new SOTA models constantly.
- Today's best model for a task is often second-best next month.
- Developers want the newest thing; standardizing on one tool leaves you behind.
- No single tool stays best for long — betting the org on one is a losing bet.
- Transition → "So what do the best engineers do?"

### 3. The best engineers are using multiple tools
- They pick the right tool *and* model per step, not one tool for everything.
- Plan with Claude Code + Opus, build with Sonnet, review with Codex + GPT-5.5.
- Same MCPs and skills (Jira, GitHub, Devbox) across every step.
- A real productivity multiplier — but only if the tools work together.
- Transition → "And here's the catch: they don't."

### 4. Configuring each agent is challenging
- Every agent is its own silo: different config file, format, and location.
- Claude `.mcp.json`/`CLAUDE.md`, Codex `config.toml`/`AGENTS.md`, Gemini `settings.json`/`GEMINI.md`.
- Same MCP server, wired three different ways; auth is unique to each tool.
- Nothing's portable — set it up once, redo it in the next agent, times every developer.
- Transition → "And it's not just annoying — it's risky."

### 5. Each agent is your most privileged user
- Via MCP, your agent reaches repos, Slack, Jira, docs, cloud, CI — everything you can.
- The agent is effectively your most privileged user; that reach is the value *and* the danger.
- Each silo'd tool holds its own copy of credentials to all of it.
- One bad tool call drops a cluster or messages the wrong person.
- (Fold in the admin angle:) for admins this also means N contracts, N consoles, no central visibility or policy.
- Transition → "So orgs get stuck on one of two islands."

### 6. Two islands
- **Locked down:** one approved tool, slow approvals, behind the frontier, lost productivity.
- **Wild west:** every tool, limited guardrails, token-maxing, `--dangerously-skip-permissions`, one call from disaster.
- Both are bad — one kills productivity, the other invites catastrophe.
- (Fold in the thesis:) you shouldn't have to choose — developers want freedom, admins want control, and you can have both.
- Transition → "Here's how we close that gap."

## Solution (~2 min)

### 7. Omnigent
- Meet **Omnigent** — the meta coding agent — on **Unity AI Gateway**.
- Omnigent is the developer CLI; Unity AI Gateway is the governance backend.
- One line: freedom for developers, visibility for admins — the bridge between the two islands.
- Transition → "Here's how it fits together."

### 8. One path for every request
- Omnigent wraps every harness; every request flows through Unity AI Gateway.
- The gateway gives you models (Databricks-hosted + external: Claude, GPT, Gemini, Kimi), the MCP catalog (built-in + external), and governed traces/usage in Unity Catalog (open format).
- One chokepoint = total visibility without limiting developer choice.
- Devs free on the left, admins see everything on the right — same pipe.
- Transition → "Let's start with the developer side."

### 9. Omnigent gives developers freedom
- One CLI, every harness, one login.
- Same plan/build/review, one command each: `omni claude --model opus`, `omni claude --model sonnet`, `omni codex`, `omni gemini`.
- MCPs, skills, and config follow you — set up once, work everywhere.
- Transition → "Let me just show you."

## Demo (~5 min — the heart of the talk)

### 10. Developer workflow (LIVE — ~3 min)
- Multi-device: start a task, continue it in Omnigent elsewhere.
- Switch agents mid-task — MCPs and skills carry over, no reconfiguring.
- Show your own usage (tokens/cost) and a contextual policy adapting to the repo.
- Narrate: "Never left my flow, never set anything up twice."

### 11. Admin workflow (LIVE — ~2 min)
- Flip to the admin view — everything you just did flowed through the gateway.
- Traces of real requests; usage by user/model/tool.
- Inference tables + dashboards in Unity Catalog; budgets and rate limits.
- Takeaway: full visibility and control, zero developer friction.
- Transition → "This isn't hypothetical — we run it ourselves."

## Proof + close (~4 min)

### 12. How Databricks runs on it
- We run this across all of Databricks engineering — 2,200 engineers, one gateway, every agent.
- The dogfooding story: we feel our own pain and gains.

### 13. The numbers
- 2,200+ engineers, 25K+ commits/month, 15K+ deployments/month, 25M+ LOC, 7 languages.
- Real, heavy load — not a pilot.

### 14. Three paths to success
- **Measure everything**, **everyone leaning in**, **move fast.**
- Tee up the next two slides as evidence of the first two.

### 15. We measure everything
- Our real internal dashboard — 90%+ weekly usage, power users ~88%, thousands of weekly active engineers and climbing.
- The curve is the proof: good + visible experience makes adoption compound.

### 16. Everyone needs to be leaning in — including your CEO
- Adoption needs top-down signal. The punchline: even Ali ships PRs through this (the Isaac model-selection PR).
- When leadership is in the tool, the org follows.

### 17. Bottlenecks are shifting
- When everyone generates code fast, the constraint moves downstream: code reviews, scaling CI, testing.
- Honest note on where the next work is.
- Transition → "So if you want this…"

### 18. Try it
- CTA: **Omnigent + Unity AI Gateway** — governance/observability/privacy for admins, faster iteration for devs.
- Not on Databricks? **Neon AI Gateway.**
- End on the dual promise: freedom *and* governance, no trade-off. → Q&A.
