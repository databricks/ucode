# Test instructions for Claude Code

Read and follow [AGENTS.md](AGENTS.md) in this directory and the coverage matrix
in [README.md](README.md) before changing tests. Those are the shared instructions
for adding, modifying and removing tests; do not maintain a different policy here.

In particular, `integration/` uses the real installed ug CLI, selected real agent
versions, and the real e2e workspace. No mocks, monkeypatching, internal application
calls, fake binaries/services, fabricated ug state, or test-only production
behavior. Do not hide failures with skips, xfails, retries or weaker assertions.
Update the matrix when coverage changes and distinguish automated coverage from
gaps and checks that were not executed.
