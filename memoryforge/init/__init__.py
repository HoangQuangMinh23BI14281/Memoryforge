"""Project bootstrap and hook ingestion."""

from memoryforge.init.bootstrap import init_project
from memoryforge.init.hooks import handle_hook_event

__all__ = ["handle_hook_event", "init_project"]
