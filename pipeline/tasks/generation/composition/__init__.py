"""
Composition task.

Weaves excerpts into coherent synthetic text.
"""

from .model import CompositionOutput
from .prompt import composition_prompt
from .task import CompositionTask

__all__ = [
    "CompositionTask",
    "CompositionOutput",
    "composition_prompt",
]
