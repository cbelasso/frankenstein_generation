"""
Task system for pipeline operations.

Tasks define input/output shapes and prompt construction.
Auto-registration via __init_subclass__.
"""

from .base import BaseTask, get_all_tasks, get_task, list_tasks
from .models import TextInput
from .runner import run_task

__all__ = [
    "BaseTask",
    "get_task",
    "list_tasks",
    "get_all_tasks",
    "TextInput",
    "run_task",
]
