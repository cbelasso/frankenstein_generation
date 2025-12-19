"""
Alerts facet definition.

Auto-registers the alerts facet with the registry.
"""

from pipeline.capabilities.classification.registry import BaseFacet

from .model import AlertsOutput
from .prompt import alert_detection_prompt


class AlertsFacet(BaseFacet):
    """Facet for detecting workplace alerts and safety concerns."""

    name = "alerts"
    output_model = AlertsOutput
    prompt_fn = alert_detection_prompt
