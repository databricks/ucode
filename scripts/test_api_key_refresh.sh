#!/usr/bin/env bash
# Test whether Claude Code handles apiKeyHelper failures gracefully.
#
# Runs two scenarios back-to-back:
#   Mode A (empty):   helper returns "" with exit 0  → hangs silently (known bad behavior)
#   Mode B (nonzero): helper exits 1 with stderr msg → should surface an error and exit
#
# Usage: bash scripts/test_api_key_refresh.sh

set -euo pipefail

SETTINGS="$HOME/.claude/settings.json"
BACKUP="$HOME/.claude/settings.json.test_backup"
CALL_COUNT_FILE="/tmp/fake_api_key_helper_call_count"
LOG_FILE="/tmp/fake_api_key_helper.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/fake_api_key_helper.sh"
TIMEOUT=30  # seconds before giving up on a hung claude invocation

cleanup() {
    if [[ -f "$BACKUP" ]]; then
        echo ""
        echo "--- Restoring original settings.json ---"
        cp "$BACKUP" "$SETTINGS"
        rm "$BACKUP"
    fi
}
trap cleanup EXIT

chmod +x "$HELPER"

patch_settings() {
    python3 - "$SETTINGS" "$HELPER" <<'EOF'
import json, sys
path, helper = sys.argv[1], sys.argv[2]
with open(path) as f:
    s = json.load(f)
s["apiKeyHelper"] = helper
s.setdefault("env", {})["CLAUDE_CODE_API_KEY_HELPER_TTL_MS"] = "5000"
with open(path, "w") as f:
    json.dump(s, f, indent=2)
EOF
}

run_scenario() {
    local mode="$1"
    echo ""
    echo "======================================================"
    echo " SCENARIO: failure mode = $mode"
    echo "======================================================"

    # Reset call counter and log
    rm -f "$CALL_COUNT_FILE" "$LOG_FILE"

    # Backup and patch settings
    cp "$SETTINGS" "$BACKUP"
    patch_settings

    echo ""
    echo "--- Run 1: first invocation (real token) ---"
    FAKE_HELPER_FAILURE_MODE="$mode" \
        timeout "$TIMEOUT" claude --dangerously-skip-permissions -p "Reply with only: HELLO_ONE" 2>&1 \
        | tail -5 || echo "(exit code: $?)"

    echo ""
    echo "--- Waiting 8s for TTL to expire ---"
    sleep 8

    echo ""
    echo "--- Run 2: second invocation (helper will fail with mode=$mode) ---"
    FAKE_HELPER_FAILURE_MODE="$mode" \
        timeout "$TIMEOUT" claude --dangerously-skip-permissions -p "Reply with only: HELLO_TWO" 2>&1 \
        | tail -10 \
        && echo "(exited 0)" || echo "(exited non-zero / timed out after ${TIMEOUT}s)"

    echo ""
    echo "--- Helper log ---"
    if [[ -f "$LOG_FILE" ]]; then
        cat "$LOG_FILE"
    else
        echo "(helper was never called)"
    fi
    echo "Total helper calls: $(cat "$CALL_COUNT_FILE" 2>/dev/null || echo 0)"

    # Restore settings before next scenario
    cp "$BACKUP" "$SETTINGS"
    rm "$BACKUP"
}

echo "=== Claude Code apiKeyHelper failure mode comparison ==="
echo "Helper: $HELPER"

run_scenario "empty"
run_scenario "nonzero"
run_scenario "reauth"

echo ""
echo "=== Complete. Compare the three scenarios above. ==="
echo ""
echo "Expected results:"
echo "  empty   — retries forever, times out (broken)"
echo "  nonzero — retries forever, times out (broken)"
echo "  reauth  — recovers on second call, returns HELLO_TWO (good)"
