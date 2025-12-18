"""Prompt registry and loader."""

from typing import Callable, Dict, Union

from .base import PromptTemplate, format_prompt
from .alerts import EXTRACTION_PROMPT as ALERTS_EXTRACTION_PROMPT, format_extraction_prompt
from .generation import (
    FRANKENSTEIN_PROMPT,
    STYLE_TRANSFER_PROMPT,
    NON_ALERT_PROMPT,
    NON_ALERT_CLASSIFICATION_DESCRIPTIONS,
    format_frankenstein_prompt,
    format_style_transfer_prompt,
    format_non_alert_prompt,
)


# Prompt registry - maps string names to PromptTemplate instances
PROMPT_REGISTRY: Dict[str, PromptTemplate] = {
    # Alerts
    "alerts.EXTRACTION_PROMPT": ALERTS_EXTRACTION_PROMPT,
    # Generation
    "generation.FRANKENSTEIN_PROMPT": FRANKENSTEIN_PROMPT,
    "generation.STYLE_TRANSFER_PROMPT": STYLE_TRANSFER_PROMPT,
    "generation.NON_ALERT_PROMPT": NON_ALERT_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """Get a prompt template by name."""
    if name not in PROMPT_REGISTRY:
        available = list(PROMPT_REGISTRY.keys())
        raise ValueError(f"Unknown prompt: {name}. Available: {available}")
    return PROMPT_REGISTRY[name]


def register_prompt(name: str, prompt: PromptTemplate) -> None:
    """Register a new prompt template."""
    PROMPT_REGISTRY[name] = prompt


__all__ = [
    # Base
    "PromptTemplate",
    "format_prompt",
    # Alerts
    "ALERTS_EXTRACTION_PROMPT",
    "format_extraction_prompt",
    # Generation
    "FRANKENSTEIN_PROMPT",
    "STYLE_TRANSFER_PROMPT",
    "NON_ALERT_PROMPT",
    "NON_ALERT_CLASSIFICATION_DESCRIPTIONS",
    "format_frankenstein_prompt",
    "format_style_transfer_prompt",
    "format_non_alert_prompt",
    # Registry
    "PROMPT_REGISTRY",
    "get_prompt",
    "register_prompt",
]
