# MemoryForge CLI

```bash
memoryforge --db ~/.memoryforge/memory.db search \
  --agent-id agent \
  --query SQLite

memoryforge --db ~/.memoryforge/memory.db recall-memory \
  --agent-id agent \
  --query "Alice SQLite"

memoryforge --db ~/.memoryforge/memory.db rlm-load README.md \
  --agent-id agent

memoryforge --db ~/.memoryforge/memory.db dispatch \
  --agent-id agent \
  --query SQLite
```

MemoryForge is distributed through PyPI, but the supported user-facing surface
is the Codex CLI/MCP workflow. Internal modules are used by the CLI, MCP server,
tests, and benchmarks, but they are not the documented integration surface for
users.
