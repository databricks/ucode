1. ALWAYS perform comprehensive live testing of every single feature, fix, or refactor you implement.

2. PROVE every new gate, seal, contract, or guard test can catch a real defect: after it
   passes, show it FAILING against a deliberately wrong-but-*present* implementation — a
   near-miss such as a flipped condition, an off-by-one, or the wrong-but-same-shaped value
   — not merely an absent or stubbed one. A test that stays green against a plausible wrong
   implementation is a proxy assertion; rewrite it until the near-miss goes red. (Red against
   a stub proves only that the test fails when the code is missing, never that it fails when
   the code is present and wrong — that gap is where weak assertions hide.)

3. NEVER assert a stand-in for the property under test. On structured artifacts (YAML, JSON,
   config, generated files) a substring check (`toContain`, `indexOf`), a bare count, or a
   value coupled to a moving target is a proxy that passes for the wrong reason — e.g.
   `toContain('V-1')` is also satisfied by `V-10`. Anchor to the exact property with a
   structural or exact-equality assertion that matches it and nothing else.

4. For every "must fail loud" criterion — a seal gate, a deny rule, a validation boundary —
   ALWAYS write an explicit BREACH test that drives the failure path and asserts the exact
   error, not only the happy-path success.

5. NEVER hand-maintain the file set a source-scan security gate inspects (the no-eval /
   no-persistence / no-network grep tests). Derive it from a glob over the package's runtime
   source (e.g. `src/**/*.ts` excluding `*.test.*` and type-only re-export barrels), by rule
   not by manual omission — a literal file array silently under-covers a new or moved module
   and over-couples the gate to unrelated stubs owned by later work. Within the scan, strip
   comments and string literals before matching and anchor each forbidden token to a
   call/word boundary (e.g. `\beval\s*\(`) so the count is a capability check, not a
   substring a comment can satisfy.

6. ALWAYS exercise a source-scan security gate (no-eval / no-persistence / no-network)
   through its OWN exported gate artifacts — the single forbidden-pattern array and the
   glob-derived file set — never through a private re-declared copy or an inline literal.
   (a) Its near-miss/breach tests MUST iterate the actual exported pattern array and, for
   every branch of each token alternation, seed one real matching call and assert that
   exact branch matches exactly once, so a typo in any single branch (e.g. `\bsendBeacon\b`
   → `\bsendBecon\b`) turns a test RED. (b) The glob's exclusion of barrels MUST anchor to
   exact top-level source-relative paths (`== 'index.ts'` / `== 'internal.ts'`), never a
   basename match at any depth, and that anchoring MUST have its own unit assertion proving
   a nested same-named module (`builders/index.ts`) is NOT excluded. A gate whose breach
   test verifies a private regex copy, or whose file set silently under-covers via an
   over-broad exclusion, passes for the wrong reason and cannot catch a real capability
   regression (extends rules 2, 3, 5).

7. ALWAYS cover the WIRING POINT, not only the extracted unit. When a story moves
   behavior into a factory or helper and unit-tests it in isolation, the call site
   that wires it in — a provider fallback (`x ?? createDefaultTelemetrySink()`), a
   dispose/cleanup effect, or a new context field — is itself a deliverable and MUST
   carry a test at that integration point that goes RED when the wiring is reverted or
   deleted: mount the real consumer WITHOUT the injected dependency, assert the
   resolved dependency's observable behavior, then tear down and assert release.
   Unit-testing the extracted unit alone does NOT satisfy this. Corollary: when a
   story swaps a default or fallback, grep existing suites for tests that describe the
   OLD behavior and rewrite them — an assertion like `emit(event) === undefined` that
   BOTH the removed and the new implementation satisfy is a proxy (rule 3) that now
   passes for the wrong reason; fix it and add every touched test file to the story
   File List (extends rules 2, 3).

8. VERIFY rendered CSS behavior through the CSSOM, never through an inline-style string.
   When a test asserts a keyframe animation, an `@media (prefers-reduced-motion: reduce)`
   exemption, or any cascade-applied value, it MUST read the parsed rule from
   `document.styleSheets` — assert a `CSSKeyframesRule` by name, or the `CSSMediaRule`
   selector→declaration — never assert only an element's inline `style.animation*` /
   `style.*`. jsdom's `getComputedStyle` applies neither stylesheet cascade nor `@media`,
   so an inline-style assertion passes identically whether or not the backing `@keyframes`
   / `@media` rule shipped, and stays green against a component that omits the rule
   entirely (a proxy, rules 2-3). Before landing the test, name the wrong-but-present
   implementation it must turn RED — a missing keyframe, a blanket rule that also zeros the
   exempt animation, a hardcoded duration; if you cannot name one, rewrite it. And NEVER
   assert a negative (zero `<tbody>` rows, no export button) against a component that
   structurally cannot produce the positive — the assertion is vacuously true; place it at
   the integration point (the dispatch, the fallback-table renderer) where the affordance
   could actually appear (extends rules 2, 3).