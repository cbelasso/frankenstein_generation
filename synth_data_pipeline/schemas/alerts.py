"""Alerts extraction schemas."""

from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel

from .base import ExtractionOutput


ALERT_TYPES = Literal[
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

SEVERITY_LEVELS = Literal["low", "moderate", "high", "critical"]

NON_ALERT_CLASSIFICATIONS = Literal[
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


class AlertSpan(BaseModel):
    """A single alert extracted from text."""
    
    excerpt: str
    reasoning: str
    alert_type: ALERT_TYPES
    severity: SEVERITY_LEVELS


class AlertsOutput(ExtractionOutput):
    """Output schema for alerts extraction."""
    
    has_alerts: bool
    alerts: List[AlertSpan]
    non_alert_classification: Optional[NON_ALERT_CLASSIFICATIONS] = None
    non_alert_reasoning: Optional[str] = None
    
    def get_labels(self) -> Set[str]:
        return {a.alert_type for a in self.alerts} if self.alerts else set()
    
    def get_excerpts(self) -> List[Dict[str, Any]]:
        return [
            {
                "excerpt": a.excerpt,
                "reasoning": a.reasoning,
                "label": a.alert_type,
                "severity": a.severity,
            }
            for a in self.alerts
        ]
    
    def has_positive_labels(self) -> bool:
        return self.has_alerts
    
    def get_negative_classification(self) -> Optional[str]:
        return self.non_alert_classification
