"""Configuration loading and management."""

from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml
from pydantic import BaseModel

from .schemas import get_schema
from .prompts import get_prompt


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str) -> Dict:
    """Load a YAML file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str, overrides: Optional[Dict] = None) -> Dict:
    """Load config with inheritance and CLI overrides."""
    config_path = Path(config_path)
    config = load_yaml(config_path)
    
    # Handle inheritance
    if "_base_" in config:
        base_path = config_path.parent / config.pop("_base_")
        base_config = load_yaml(base_path)
        config = deep_merge(base_config, config)
    
    # Apply CLI overrides
    if overrides:
        config = deep_merge(config, overrides)
    
    return config


class PipelineConfig:
    """Configuration wrapper with convenience methods."""
    
    def __init__(self, config: Dict):
        self.config = config
    
    @classmethod
    def from_file(cls, path: str, overrides: Optional[Dict] = None) -> "PipelineConfig":
        """Load config from file."""
        return cls(load_config(path, overrides))
    
    @classmethod
    def from_dict(cls, config: Dict) -> "PipelineConfig":
        """Create config from dictionary."""
        return cls(config)
    
    # Input config
    @property
    def input_file(self) -> Optional[str]:
        return self.config.get("input", {}).get("file")
    
    @property
    def input_column(self) -> Optional[str]:
        return self.config.get("input", {}).get("column")
    
    @property
    def max_samples(self) -> Optional[int]:
        return self.config.get("input", {}).get("max_samples")
    
    # Processor config
    @property
    def processor_config(self) -> Dict:
        return self.config.get("processor", {})
    
    @property
    def model(self) -> str:
        return self.processor_config.get("model", "casperhansen/mistral-nemo-instruct-2407-awq")
    
    @property
    def gpus(self) -> list:
        return self.processor_config.get("gpus", [0])
    
    @property
    def batch_size(self) -> int:
        return self.processor_config.get("batch_size", 25)
    
    @property
    def gpu_memory_utilization(self) -> float:
        return self.processor_config.get("gpu_memory_utilization", 0.9)
    
    @property
    def max_model_len(self) -> int:
        return self.processor_config.get("max_model_len", 4096)
    
    @property
    def multiplicity(self) -> int:
        return self.processor_config.get("multiplicity", 1)
    
    # Extraction config
    @property
    def extraction_config(self) -> Dict:
        return self.config.get("extraction", {})
    
    @property
    def extraction_schema_name(self) -> str:
        return self.extraction_config.get("schema", "alerts.AlertsOutput")
    
    @property
    def extraction_prompt_name(self) -> str:
        return self.extraction_config.get("prompt", "alerts.EXTRACTION_PROMPT")
    
    @property
    def extraction_guided_config(self) -> Dict:
        return self.extraction_config.get("guided_config", {})
    
    def get_extraction_schema(self) -> Type[BaseModel]:
        return get_schema(self.extraction_schema_name)
    
    def get_extraction_prompt(self):
        return get_prompt(self.extraction_prompt_name)
    
    # Generation config
    @property
    def generation_config(self) -> Dict:
        return self.config.get("generation", {})
    
    @property
    def generation_enabled(self) -> bool:
        return self.generation_config.get("enabled", True)
    
    @property
    def generation_type(self) -> str:
        return self.generation_config.get("type", "alerts")
    
    @property
    def n_samples(self) -> int:
        return self.generation_config.get("n_samples", 5)
    
    @property
    def n_candidates(self) -> int:
        return self.generation_config.get("n_candidates", 5)
    
    @property
    def generation_guided_config(self) -> Dict:
        return self.generation_config.get("guided_config", {})
    
    @property
    def target_labels(self) -> Optional[list]:
        return self.generation_config.get("alert", {}).get("target_labels")
    
    @property
    def target_classifications(self) -> Optional[list]:
        return self.generation_config.get("non_alert", {}).get("target_classifications")
    
    # Validation config
    @property
    def validation_config(self) -> Dict:
        return self.config.get("validation", {})
    
    @property
    def validation_enabled(self) -> bool:
        return self.validation_config.get("enabled", True)
    
    @property
    def match_threshold(self) -> float:
        return self.validation_config.get("match_threshold", 0.5)
    
    # Output config
    @property
    def output_dir(self) -> str:
        return self.config.get("output", {}).get("dir", "./output")
    
    # Task info
    @property
    def task_name(self) -> str:
        return self.config.get("task", {}).get("name", "unknown")
    
    @property
    def task_description(self) -> str:
        return self.config.get("task", {}).get("description", "")
    
    def __repr__(self) -> str:
        return f"PipelineConfig(task={self.task_name}, model={self.model})"
