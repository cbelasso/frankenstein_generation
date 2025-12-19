"""
Alert detection facet.

Detects serious workplace concerns that require immediate attention.
"""

from .facet import AlertsFacet
from .model import AlertsOutput, AlertSpan
from .prompt import alert_detection_prompt

__all__ = [
    "AlertsFacet",
    "AlertsOutput",
    "AlertSpan",
    "alert_detection_prompt",
]
