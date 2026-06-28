"""Project bootstrap and hook ingestion."""

from memoryforge.init.bootstrap import ensure_project_initialized, init_project
from memoryforge.init.hooks import handle_hook_event

__all__ = ["ensure_project_initialized", "handle_hook_event", "init_project"]
