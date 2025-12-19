"""
Alert detection models.

Defines the schema for alert detection output.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel


class AlertSpan(BaseModel):
    """A single detected alert span."""

    excerpt: str
    reasoning: str
    alert_type: Literal[
        "discrimination",
        "sexual_harassment",
        "severe_harassment",
        "bullying",
        "workplace_violence",
        "threat_of_violence",
        "coercive_threat",
        "safety_hazard",
        "retaliation",
        "substance_abuse_at_work",
        "data_breach",
        "security_incident",
        "fraud",
        "corruption",
        "quid_pro_quo",
        "ethics_violation",
        "mental_health_crisis",
        "pattern_of_unfair_treatment",
        "workload_burnout_risk",
        "management_concern",
        "interpersonal_conflict",
        "professional_misconduct",
        "inappropriate_language",
        "profanity",
        "suggestive_language",
        "mental_wellbeing_concern",
        "physical_safety_concern",
    ]
    severity: Literal["low", "moderate", "high", "critical"]


class AlertsOutput(BaseModel):
    """Output schema for alert detection."""

    has_alerts: bool
    alerts: List[AlertSpan] = []
    non_alert_classification: Optional[
        Literal[
            "performance_complaint",
            "quality_complaint",
            "workload_feedback",
            "process_improvement",
            "resource_request",
            "general_dissatisfaction",
            "constructive_feedback",
            "positive_feedback",
            "neutral_comment",
            "unclear",
        ]
    ] = None
    non_alert_reasoning: Optional[str] = None
