"""
Recommendation detection facet.

Extracts actionable suggestions and proposals from employee comments.
"""

from .facet import RecommendationsFacet
from .model import RecommendationsOutput, RecommendationSpan
from .prompt import recommendations_detection_prompt

__all__ = [
    "RecommendationsFacet",
    "RecommendationsOutput",
    "RecommendationSpan",
    "recommendations_detection_prompt",
]
