"""Synthetic data generation pipeline."""

# Config
from .config import PipelineConfig, load_config

# Schemas
from .schemas import (
    ExtractionOutput,
    GeneratedOutput,
    AlertSpan,
    AlertsOutput,
    GeneratedComment,
    GeneratedNonAlertComment,
    StyleTransferOutput,
    SCHEMA_REGISTRY,
    get_schema,
    register_schema,
)

# Prompts
from .prompts import (
    PromptTemplate,
    ALERTS_EXTRACTION_PROMPT,
    FRANKENSTEIN_PROMPT,
    STYLE_TRANSFER_PROMPT,
    NON_ALERT_PROMPT,
    PROMPT_REGISTRY,
    get_prompt,
    register_prompt,
    format_frankenstein_prompt,
    format_style_transfer_prompt,
    format_non_alert_prompt,
)

# Core
from .core import (
    DataStore,
    Extractor,
    AlertsExtractor,
    create_extractor,
    Generator,
    BalancedGenerator,
    Validator,
    ValidationResult,
    print_validation_result,
)


__all__ = [
    # Config
    "PipelineConfig",
    "load_config",
    # Schemas
    "ExtractionOutput",
    "GeneratedOutput",
    "AlertSpan",
    "AlertsOutput",
    "GeneratedComment",
    "GeneratedNonAlertComment",
    "StyleTransferOutput",
    "SCHEMA_REGISTRY",
    "get_schema",
    "register_schema",
    # Prompts
    "PromptTemplate",
    "ALERTS_EXTRACTION_PROMPT",
    "FRANKENSTEIN_PROMPT",
    "STYLE_TRANSFER_PROMPT",
    "NON_ALERT_PROMPT",
    "PROMPT_REGISTRY",
    "get_prompt",
    "register_prompt",
    "format_frankenstein_prompt",
    "format_style_transfer_prompt",
    "format_non_alert_prompt",
    # Core
    "DataStore",
    "Extractor",
    "AlertsExtractor",
    "create_extractor",
    "Generator",
    "BalancedGenerator",
    "Validator",
    "ValidationResult",
    "print_validation_result",
]
