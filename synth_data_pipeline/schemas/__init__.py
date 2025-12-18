"""Schema registry and loader."""

from typing import Dict, Type

from pydantic import BaseModel

from .base import ExtractionOutput, GeneratedOutput
from .alerts import AlertSpan, AlertsOutput, ALERT_TYPES, SEVERITY_LEVELS, NON_ALERT_CLASSIFICATIONS
from .generated import GeneratedComment, GeneratedNonAlertComment, StyleTransferOutput


# Schema registry - maps string names to classes
SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    # Alerts
    "alerts.AlertSpan": AlertSpan,
    "alerts.AlertsOutput": AlertsOutput,
    # Generated
    "generated.GeneratedComment": GeneratedComment,
    "generated.GeneratedNonAlertComment": GeneratedNonAlertComment,
    "generated.StyleTransferOutput": StyleTransferOutput,
}


def get_schema(name: str) -> Type[BaseModel]:
    """Get a schema class by name."""
    if name not in SCHEMA_REGISTRY:
        available = list(SCHEMA_REGISTRY.keys())
        raise ValueError(f"Unknown schema: {name}. Available: {available}")
    return SCHEMA_REGISTRY[name]


def register_schema(name: str, schema: Type[BaseModel]) -> None:
    """Register a new schema."""
    SCHEMA_REGISTRY[name] = schema


__all__ = [
    # Base
    "ExtractionOutput",
    "GeneratedOutput",
    # Alerts
    "AlertSpan",
    "AlertsOutput",
    "ALERT_TYPES",
    "SEVERITY_LEVELS",
    "NON_ALERT_CLASSIFICATIONS",
    # Generated
    "GeneratedComment",
    "GeneratedNonAlertComment",
    "StyleTransferOutput",
    # Registry
    "SCHEMA_REGISTRY",
    "get_schema",
    "register_schema",
]
