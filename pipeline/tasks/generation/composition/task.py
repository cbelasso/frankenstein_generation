"""
Composition task definition.

Auto-registers the composition task with the registry.
"""

from pipeline.tasks.base import BaseTask
from pipeline.tasks.generation.models import ExcerptSet

from .model import CompositionOutput
from .prompt import composition_prompt


class CompositionTask(BaseTask):
    """Task for composing excerpts into coherent synthetic text."""

    name = "composition"
    category = "generation"
    input_model = ExcerptSet
    output_model = CompositionOutput
    prompt_fn = composition_prompt
    default_config = {
        "temperature": 0.7,  # Higher for more creative variation
        "max_tokens": 500,
    }
