#!/usr/bin/env bash
set -euo pipefail

MEMORYFORGE_DB="${MEMORYFORGE_DB:-$HOME/.memoryforge/memory.db}"
AGENT_ID="${AGENT_ID:-default}"

if command -v memoryforge >/dev/null 2>&1; then
  memoryforge --db "$MEMORYFORGE_DB" graph-search --agent-id "$AGENT_ID" --query "session context" --limit 5 >/dev/null || true
fi
