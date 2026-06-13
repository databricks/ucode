# Talk Track — Governing Coding Agents at Enterprise Scale (DAIS)

**Length:** ~40 min · **Speakers:** Rohit Agrawal + Aarushi Shah · **Slides:** 21

**Suggested split:** Rohit runs slides 1–13 (setup → solution → developer demo); Aarushi
takes 14–21 (admin demo → Databricks case study → close). Handoff happens on slide 14.

**Through-line:** The frontier moves every week, so the winning move is using *many*
agents. But that creates two problems — a developer portability problem and an admin
governance problem. **Omnigent** (developer freedom) on **Unity AI Gateway** (admin
governance) solves both, and Databricks runs on it across 2,200 engineers.

---

## Setup — Rohit (~10 min)

### 1. Title — "Governing Coding Agents at Enterprise Scale"
- Welcome and quick intros — both SWEs on Databricks AI; we build the coding-agent platform and live in it daily.
- Frame the whole talk: two audiences, one problem. Developers want freedom to use any agent; admins need governance.
- Promise: a live demo of *both* the developer and admin experience.
- Promise: we'll show how Databricks itself runs this across all of engineering.
- Set the tone: this is real and in production, not a vision deck.
- Transition → "Let's start with what's actually happening in the ecosystem."

### 2. A new coding agent every week
- The pace is relentless — new agents (Claude Code, Codex, Gemini CLI, Cursor, Windsurf…) and new SOTA models land constantly.
- The frontier genuinely moves week to week; today's best model for a task is often second-best next month.
- Developers feel this and want to try the newest thing immediately.
- Standardizing on a single tool means you're permanently behind the frontier.
- No vendor wins forever — betting the whole org on one is a losing bet.
- Transition → "So what do the best engineers actually do about it?"

### 3. The best engineers are using multiple tools
- They don't pick one tool — they pick the right tool *and* model for each step of the work.
- Concrete workflow: plan with Claude Code + Opus, build with Claude Code + Sonnet, review with Codex + GPT-5.5.
- Different strengths for different jobs: deep reasoning to plan, speed to build, a second set of eyes to review.
- The same MCPs and skills (Jira, GitHub, Devbox) run across every step.
- This is a real productivity multiplier — *but only if the tools work together.*
- Transition → "Here's the problem: they don't."

### 4. Configuring each agent is challenging
- Each agent is its own silo — different config file, different format, different location.
- Claude uses `.mcp.json` + `CLAUDE.md`; Codex uses `config.toml` + `AGENTS.md`; Gemini uses `settings.json` + `GEMINI.md`.
- The *same* MCP server has to be wired up three different ways.
- The worst part: auth is unique to each — a separate login and separate credentials per tool.
- Nothing is portable; you set it up once, then redo it all in the next agent.
- Multiply that by every developer and it's real, invisible, daily friction.
- Transition → "And it's not just friction — it's risk."

### 5. Each agent is your most privileged user
- Through MCP, your agent reaches your repos, Slack, Jira, company docs, cloud infra, and CI/CD.
- That's the same access *you* have — the agent is effectively your most privileged user.
- The reach is exactly what makes agents useful and what makes them dangerous.
- Each of those silo'd tools is holding its own copy of credentials to all of it.
- One bad tool call can drop a cluster or message the wrong person.
- Transition → "For developers that's friction and risk. For admins, it's a different nightmare."

### 6. Deploying coding agents is a nightmare for admins
- Every new agent multiplies what the admin has to manage.
- Multiple vendor contracts, multiple admin consoles, fragmented billing across teams.
- No centralized usage visibility — who's using what, and what it costs.
- Hard to enforce security and policy consistently across every tool.
- So admins default to "no," simply because they can't see or control it.
- Transition → "Which leaves organizations stuck on one of two islands."

### 7. Two islands
- Island 1 — **Locked down:** one approved tool, slow approvals, stuck behind the frontier, lost productivity, shadow IT creeps in.
- Island 2 — **Wild west:** every tool allowed, limited guardrails, token-maxing, `--dangerously-skip-permissions`, one call from disaster.
- Most orgs are sitting on one of these right now.
- Both are bad: one kills productivity, the other invites catastrophe.
- Transition → "The real question is whether you actually have to choose."

### 8. How do we bridge the gap?
- State the thesis plainly: you shouldn't have to trade freedom for safety.
- The goal — give developers full freedom *and* admins full control, at the same time.
- That's the bridge we set out to build. (Keep this slide short; it's a pivot.)
- Transition → "Here's what both sides actually want."

### 9. What everyone actually wants
- Developers want: choice of agent and model, MCPs/skills that follow them, freedom to move fast without fear.
- Admins want: cost controls, governance, observability, privacy guarantees.
- These aren't opposing demands — they compose into a single outcome.
- The trick is an architecture that serves both without compromise.
- Transition → "Let me introduce it."

## Solution — Rohit (~6 min)

### 10. Omnigent
- Meet **Omnigent** — the meta coding agent — running on **Unity AI Gateway**.
- Omnigent is the developer-facing CLI; Unity AI Gateway is the governance backend.
- One line: *freedom for developers, visibility for admins.*
- This is the bridge between the two islands.
- Transition → "Here's how it fits together."

### 11. One path for every request
- Omnigent wraps every harness — Claude Code, Codex, Gemini, and more.
- Every request it makes flows through Unity AI Gateway — that's the key design choice.
- The gateway provides: models (Databricks-hosted + external — Claude, GPT, Gemini, Kimi), the MCP catalog (built-in + external), and governed tables in Unity Catalog (traces + usage, open format).
- One chokepoint = total visibility *without* limiting developer choice.
- Developers work freely on the left; admins get full visibility on the right; same pipe.
- Transition → "Let's look at the developer side first."

### 12. Omnigent gives developers freedom
- One CLI wraps every harness; one login for all of them.
- Same plan/build/review, now one command each: `omni claude --model opus` (plan), `omni claude --model sonnet` (build), `omni codex` (review), `omni gemini` (search).
- Your MCPs, skills, and config follow you across every agent — set up once, work everywhere.
- Always get the latest SOTA models without waiting on approvals.
- Transition → "Rather than talk about it, let me show you."

## Developer demo — Rohit, LIVE (~8 min)

### 13. Developer workflow
- Multi-device: start a task in one place, pick it up in Omnigent elsewhere.
- Switch agents mid-task — MCPs and skills carry over, no reconfiguring.
- Show your own usage (tokens / cost).
- Show a contextual policy adapting to the repo.
- Narrate the feeling: "I never left my flow, and I never set anything up twice."
- Weave in best-practices tips as you go.
- **Handoff** → "That's the developer side. Aarushi will show you what the admin sees while all of this is happening."

## Admin demo + case study — Aarushi (~16 min)

### 14. Admin workflow (LIVE)
- Open: everything Rohit just did flowed through the gateway — here's the admin view.
- Show traces — every request, fully traced.
- Show usage analytics — who, what, how much, across users / models / tools.
- Show inference tables + live dashboards in Unity Catalog.
- Show budgets and rate limits, per user / model / tool.
- Takeaway: full visibility and control, with zero developer friction.
- Transition → "This isn't hypothetical — we run it ourselves."

### 15. How Databricks runs on it
- Case study: Databricks runs this across *all* of engineering.
- The hook: 2,200 engineers, one gateway, every coding agent.
- This is the dogfooding story — we feel our own pain and our own gains.
- Transition → "Here's the scale."

### 16. The numbers
- 2,200+ engineers, 10+ engineering offices.
- 25K+ commits/month, 15K+ deployments/month.
- 25M+ lines of code, 7 languages.
- The point: this runs under real, heavy, heterogeneous load — not a pilot.
- Transition → "Three things made the rollout actually work."

### 17. Three paths to success
- **Measure everything** — instrument adoption, cost, and impact from day one.
- **Everyone leaning in** — ICs to the CEO, all in.
- **Move fast** — ship, learn, iterate weekly.
- Tee up the next two slides as the evidence for points 1 and 2.
- Transition → "First — measure everything."

### 18. We measure everything
- This is our *real* internal dashboard, built on Unity Catalog.
- 90%+ weekly usage rate; power + regular users around 88%.
- Thousands of weekly active engineers, climbing every week.
- The curve is the proof — adoption compounds when the experience is good and visible.
- Transition → "Second — everyone leaning in."

### 19. Everyone needs to be leaning in — including your CEO
- Culture point: adoption needs a top-down signal, not just bottom-up enthusiasm.
- The punchline: even Ali, our CEO, ships PRs through this — the Isaac model-selection PR.
- When leadership is in the tool, the whole org follows.
- Transition → "And as coding accelerates, the bottleneck moves."

### 20. Bottlenecks are shifting
- When everyone can generate code fast, the constraint moves downstream.
- The new bottlenecks: code reviews, scaling CI, testing.
- Honest framing — solving generation surfaces the next set of problems, and that's where we're investing.
- Transition → "So if you want this for your org…"

### 21. Try it
- CTA: **Omnigent + Unity AI Gateway** — built-in governance, observability, and privacy for admins; faster iteration for developers.
- Not a Databricks customer? **Neon AI Gateway.**
- Close on the dual promise: freedom for developers, governance for admins, no trade-off.
- Thank the room → Q&A.
