"""
Alerts task definition.

Auto-registers the alerts task with the registry.
"""

from pipeline.tasks.base import BaseTask
from pipeline.tasks.models import TextInput

from .model import AlertsOutput
from .prompt import alert_detection_prompt


class AlertsTask(BaseTask):
    """Task for detecting workplace alerts and safety concerns."""

    name = "alerts"
    category = "classification"
    input_model = TextInput
    output_model = AlertsOutput
    prompt_fn = alert_detection_prompt
    default_config = {
        "temperature": 0.1,
        "max_tokens": 1000,
    }
