"""
Recommendations task definition.

Auto-registers the recommendations task with the registry.
"""

from pipeline.tasks.base import BaseTask
from pipeline.tasks.models import TextInput

from .model import RecommendationsOutput
from .prompt import recommendations_detection_prompt


class RecommendationsTask(BaseTask):
    """Task for extracting actionable recommendations from comments."""

    name = "recommendations"
    category = "classification"
    input_model = TextInput
    output_model = RecommendationsOutput
    prompt_fn = recommendations_detection_prompt
    default_config = {
        "temperature": 0.1,
        "max_tokens": 1000,
    }
