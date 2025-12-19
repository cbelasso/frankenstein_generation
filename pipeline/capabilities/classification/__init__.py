"""
Classification capabilities package.

Provides facet-based text classification with auto-registration.

Usage:
    from pipeline.capabilities.classification import get_facet, list_facets

    # List available facets
    print(list_facets())  # ['alerts', 'recommendations']

    # Get a specific facet
    alerts_facet = get_facet("alerts")
    prompt = alerts_facet.prompt_fn("Some employee comment")
    # Use alerts_facet.output_model as schema for structured output
"""

# Import facets to trigger registration
from . import facets

# Re-export shared models
from .models import (
    ExcerptBank,
    ExcerptReference,
    ExtractedText,
    ExtractionBatch,
    ExtractionMetadata,
)

# Re-export registry functions for convenience
from .registry import BaseFacet, get_all_facets, get_facet, list_facets

__all__ = [
    # Registry
    "get_facet",
    "list_facets",
    "get_all_facets",
    "BaseFacet",
    # Models
    "ExtractionMetadata",
    "ExtractedText",
    "ExtractionBatch",
    "ExcerptReference",
    "ExcerptBank",
    # Subpackages
    "facets",
]
