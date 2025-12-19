"""
Classification facets package.

Importing this module registers all available facets with the registry.
"""

# Import facets to trigger auto-registration
from . import alerts, recommendations

__all__ = ["alerts", "recommendations"]
