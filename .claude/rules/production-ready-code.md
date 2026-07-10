# CRITICAL CONSTRAINT — ZERO TOLERANCE: Every file you touch must be left in a state of immaculate structure, naming, and clarity; dead code, duplication, unclear naming, and disorganization are treated as severity-zero defects equivalent to broken functionality — no exception, no trade-off, no shortcut.

# Rules for Production-Ready Code

## Purpose

These rules ensure maintainability, safety, and developer velocity.
**MUST** rules are non-negotiable. **SHOULD** rules are strongly recommended.

---

## Philosophy

Code is read 10x more than it is written. Every decision should optimize ruthlessly for the reader. Clean code is empathy for the next person — including future-you — expressed through structure, naming, and deletion.

- **Boy Scout Rule** — Leave every file cleaner than you found it. If you touch a file and see a stale import, a confusing name, or a dead branch, fix it in passing. This is how codebases stay clean over time instead of decaying.
- **Delete eagerly** — Commented-out code, unused variables, dead branches, and speculative utilities are rot. Git remembers; your codebase shouldn't have to. If it's not serving the current system, remove it.
- **Consistency is a first-class value** — Same problem, same solution, everywhere. If the codebase does something one way, use that way even if you know a "better" one — unless you're prepared to migrate everything. Local cleverness creates global confusion.
- **Minimal public surface area** — Expose only what consumers need. Make things private or internal by default. Every export is a contract you now maintain forever.
- **No broken windows** — Never tolerate a known mess. The moment you leave one hack in place ("we'll fix it later"), it gives permission for the next one, and decay compounds exponentially.
- **Readability over cleverness** — A one-liner using reduce with nested ternaries is worse than a five-line loop anyone can follow. If you're proud of how clever the code is, rewrite it until you're proud of how obvious it is.

---

## Planning & Approach

- **MUST** ask clarifying questions before starting complex work.
- **MUST** draft and confirm an approach before writing code. Do not code until the plan is approved.
- **SHOULD** list pros and cons when ≥ 2 viable approaches exist.
- **MUST** analyze similar parts of the codebase and ensure the plan is consistent with existing patterns, introduces minimal changes, and reuses existing code.
- **SHOULD** scope investigations narrowly. Use subagents for research to avoid polluting the main context.

---

## Code Quality

- **MUST** follow existing project conventions and patterns. Match the style, naming, and architecture already in the codebase.
- **MUST** write small, single-responsibility functions. Prefer simple, composable, testable functions over classes when classes aren't needed.
- **MUST** use domain vocabulary from the existing codebase when naming functions, variables, and types.
- **MUST** handle errors explicitly. Never swallow errors silently. Use typed errors where the language supports it.
- **MUST** validate all external inputs (API payloads, user input, environment variables) at system boundaries.
- **SHOULD** prefer pure functions over side-effecting ones where practical.
- **SHOULD NOT** add comments except for critical caveats or non-obvious "why" explanations. Write self-explanatory code instead.
- **SHOULD NOT** extract a function unless it is reused elsewhere, is the only way to unit-test otherwise untestable logic, or drastically improves readability.
- **SHOULD NOT** introduce new dependencies without justification. Prefer standard library solutions.
- **SHOULD NOT** leave `TODO`, `FIXME`, or placeholder code in production paths.

---

## Architecture & Design

- **MUST** keep changes minimal and focused. One concern per change, one purpose per PR.
- **MUST** never modify shared interfaces, schemas, or APIs without understanding downstream consumers.
- **MUST** place shared code in a shared location only if used by ≥ 2 consumers. Do not prematurely abstract.
- **SHOULD** prefer composition over inheritance.
- **SHOULD** keep coupling low between modules. Depend on interfaces/contracts, not implementations.
- **SHOULD** follow the principle of least surprise — APIs and functions should behave as their names suggest.

### KISS (Keep It Simple, Stupid)

- **MUST** choose the simplest solution that solves the actual problem. If a junior developer can't understand the code on first read, it's too complex.
- **SHOULD NOT** introduce abstractions, generics, or design patterns unless the current complexity demands them. Clever code is a liability; clear code is an asset.

### YAGNI (You Aren't Gonna Need It)

- **MUST NOT** build features, parameters, or extension points for hypothetical future requirements. Only implement what is needed right now.
- **SHOULD NOT** add configuration options, flags, or plugin architectures "just in case." Delete speculative code — it's cheaper to build later than to maintain now.

### DRY (Don't Repeat Yourself)

- **SHOULD** extract shared logic only when the same code appears in ≥ 2 places with the same intent. Duplication is far cheaper than the wrong abstraction.
- **SHOULD NOT** conflate coincidental similarity with true duplication. Two functions that happen to look alike but serve different concerns should remain separate.

---

## Testing

- **MUST** write tests for every new feature and every bug fix.
- **MUST** follow TDD when feasible: write a failing test first, then implement.
- **MUST** separate unit tests (pure logic, no I/O) from integration tests (DB, network, file system).
- **MUST** run the existing test suite after changes to confirm nothing is broken.
- **SHOULD** prefer integration tests over heavy mocking. Mocks should only replace truly external services.
- **SHOULD** test edge cases, realistic input, unexpected input, and value boundaries.
- **SHOULD** use strong assertions (`toEqual(expected)`) over weak ones (`toBeGreaterThan(0)`).
- **SHOULD** parameterize test inputs. Never embed unexplained magic numbers or literals.
- **SHOULD NOT** write trivial tests that can never fail for a real defect (e.g., `expect(true).toBe(true)`).
- **SHOULD NOT** test conditions already enforced by the type system.
- **SHOULD** ensure the test description matches exactly what the assertion verifies.

---

## Security

- **MUST** never commit secrets, API keys, tokens, or `.env` files.
- **MUST** validate and sanitize all user input before use in queries, commands, or output.
- **MUST** use parameterized queries. Never construct SQL or shell commands with string interpolation.
- **MUST** validate webhook signatures and authentication tokens on all protected endpoints.
- **SHOULD** apply the principle of least privilege for all service accounts, roles, and permissions.
- **SHOULD** review any AI-generated authentication, authorization, or payment code with extra scrutiny.

---

## Performance

- **SHOULD** be aware of N+1 queries, unnecessary re-renders, and unbounded loops.
- **SHOULD** use pagination for any endpoint that returns a list.
- **SHOULD NOT** prematurely optimize. Correctness and readability come first; optimize when profiling justifies it.

---

## Git & Version Control

- **MUST** use [Conventional Commits](https://www.conventionalcommits.org/) format for all commit messages:
  `<type>[optional scope]: <description>`
  Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`, `ci`, `build`.
- **MUST NOT** reference Claude, Anthropic, or any AI tool in commit messages.
- **SHOULD** keep commits atomic — one logical change per commit.
- **SHOULD NOT** commit generated files, build artifacts, or `node_modules`.

---

## Tooling & Verification

- **MUST** ensure the linter passes with zero errors before considering work done.
- **MUST** ensure the type checker passes with zero errors before considering work done.
- **MUST** ensure the formatter has been applied to all new or changed files.
- **MUST** run the full build to confirm compilation succeeds.
- **SHOULD** run the single relevant test file during development for speed, then the full suite before committing.

---

## Code Review Checklist (Self-Review Before Submitting)

Before considering any change complete, verify:

- [ ] Can you read each function and honestly, easily follow what it does?
- [ ] Are there unused parameters, imports, or dead code?
- [ ] Are there unnecessary type casts that could be moved to function signatures?
- [ ] Is cyclomatic complexity reasonable? (Flatten deeply nested if/else.)
- [ ] Could a well-known data structure or algorithm simplify the logic?
- [ ] Is each function name the best, most consistent choice? (Brainstorm 3 alternatives.)
- [ ] Are all new functions easily testable without mocking core infrastructure?
- [ ] Are there hidden dependencies that should be explicit parameters instead?


