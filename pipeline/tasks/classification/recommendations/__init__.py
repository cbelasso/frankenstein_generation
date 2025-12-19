"""
Recommendation detection task.

Extracts actionable suggestions and proposals from employee comments.
"""

from .model import RecommendationsOutput, RecommendationSpan
from .prompt import recommendations_detection_prompt
from .task import RecommendationsTask

__all__ = [
    "RecommendationsTask",
    "RecommendationsOutput",
    "RecommendationSpan",
    "recommendations_detection_prompt",
]
