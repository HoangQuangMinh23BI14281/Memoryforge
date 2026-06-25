#!/usr/bin/env bash
set -euo pipefail

MEMORYFORGE_DB="${MEMORYFORGE_DB:-$HOME/.memoryforge/memory.db}"
AGENT_ID="${AGENT_ID:-default}"

if command -v memoryforge >/dev/null 2>&1 && [ -n "${SESSION_FILE:-}" ]; then
  memoryforge --db "$MEMORYFORGE_DB" store-session --agent-id "$AGENT_ID" --session-file "$SESSION_FILE" || true
fi
