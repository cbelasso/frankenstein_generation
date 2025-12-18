"""
Alert detection capability.

Detects serious workplace concerns that require immediate attention.
"""

from .model import AlertsOutput, AlertSpan
from .prompt import alert_detection_prompt

__all__ = [
    "AlertsOutput",
    "AlertSpan",
    "alert_detection_prompt",
]
