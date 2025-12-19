"""
Recommendations facet definition.

Auto-registers the recommendations facet with the registry.
"""

from pipeline.capabilities.classification.registry import BaseFacet

from .model import RecommendationsOutput
from .prompt import recommendations_detection_prompt


class RecommendationsFacet(BaseFacet):
    """Facet for extracting actionable recommendations from comments."""

    name = "recommendations"
    output_model = RecommendationsOutput
    prompt_fn = recommendations_detection_prompt
