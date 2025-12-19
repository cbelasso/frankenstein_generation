"""
Alert detection task.

Detects serious workplace concerns that require immediate attention.
"""

from .model import AlertsOutput, AlertSpan
from .prompt import alert_detection_prompt
from .task import AlertsTask

__all__ = [
    "AlertsTask",
    "AlertsOutput",
    "AlertSpan",
    "alert_detection_prompt",
]
