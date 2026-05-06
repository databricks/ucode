#!/usr/bin/env bash
# Fake apiKeyHelper for testing Claude Code's token refresh behavior.
#
# First call: returns a real Databricks token.
# Subsequent calls: behavior controlled by FAKE_HELPER_FAILURE_MODE env var:
#   empty    (default) — returns empty string, exit 0
#   nonzero             — prints error to stderr, exits 1
#   reauth              — fails once then recovers (simulates re-auth fixing the problem)
#
# Each invocation is logged to /tmp/fake_api_key_helper.log with a timestamp.

CALL_COUNT_FILE="/tmp/fake_api_key_helper_call_count"
LOG_FILE="/tmp/fake_api_key_helper.log"
WORKSPACE="${DATABRICKS_HOST:-https://eng-ml-inference-team-us-east-1.cloud.databricks.com}"

# Increment call counter
if [[ -f "$CALL_COUNT_FILE" ]]; then
    count=$(cat "$CALL_COUNT_FILE")
else
    count=0
fi
count=$((count + 1))
echo "$count" > "$CALL_COUNT_FILE"

timestamp=$(date '+%Y-%m-%dT%H:%M:%S')

if [[ "$count" -eq 1 ]]; then
    # First call: get a real token
    token=$(env -u DATABRICKS_TOKEN -u DATABRICKS_CLIENT_ID -u DATABRICKS_CLIENT_SECRET \
        -u DATABRICKS_USERNAME -u DATABRICKS_PASSWORD -u DATABRICKS_AUTH_TYPE \
        databricks auth token --host "$WORKSPACE" --output json 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null)
    echo "$timestamp  call=$count  returning=real_token  len=${#token}" >> "$LOG_FILE"
    echo "$token"
else
    mode="${FAKE_HELPER_FAILURE_MODE:-empty}"
    if [[ "$mode" == "nonzero" ]]; then
        echo "$timestamp  call=$count  returning=nonzero_exit  (simulated failure)" >> "$LOG_FILE"
        echo "databricks auth: token refresh failed (simulated)" >&2
        exit 1
    elif [[ "$mode" == "reauth" ]]; then
        # Simulate: first failure triggers re-auth, subsequent calls succeed
        if [[ "$count" -eq 2 ]]; then
            # This is the "re-auth" call — takes a moment, then returns a real token
            echo "$timestamp  call=$count  returning=real_token_after_reauth  (simulated re-auth)" >> "$LOG_FILE"
        else
            echo "$timestamp  call=$count  returning=real_token  (recovered)" >> "$LOG_FILE"
        fi
        token=$(env -u DATABRICKS_TOKEN -u DATABRICKS_CLIENT_ID -u DATABRICKS_CLIENT_SECRET \
            -u DATABRICKS_USERNAME -u DATABRICKS_PASSWORD -u DATABRICKS_AUTH_TYPE \
            databricks auth token --host "$WORKSPACE" --output json 2>/dev/null \
            | python3 -c "import json,sys; print(json.load(sys.stdin).get('access_token', ''))" 2>/dev/null)
        echo "$token"
    else
        echo "$timestamp  call=$count  returning=empty  (simulated failure)" >> "$LOG_FILE"
        echo ""
    fi
fi
