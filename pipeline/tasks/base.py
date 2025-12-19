"""
Base task definition and registry.

Tasks auto-register when their class is defined by inheriting from BaseTask.
"""

from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel

_registry: Dict[str, Type["BaseTask"]] = {}


class BaseTask:
    """
    Base class for all tasks.

    Subclasses must define:
        - name: str - unique identifier for the task
        - input_model: Type[BaseModel] - Pydantic model for input validation
        - output_model: Type[BaseModel] - Pydantic model for output schema
        - prompt_fn: Callable - function that generates the prompt

    Optional:
        - category: str - grouping category (e.g., "classification", "generation")
        - default_config: dict - default sampling parameters

    Registration happens automatically via __init_subclass__.
    """

    name: str = None
    category: str = None
    input_model: Type[BaseModel] = None
    output_model: Type[BaseModel] = None
    prompt_fn: Callable[..., str] = None
    default_config: Dict[str, Any] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            if cls.input_model is None:
                raise ValueError(f"Task '{cls.name}' must define input_model")
            if cls.output_model is None:
                raise ValueError(f"Task '{cls.name}' must define output_model")
            if cls.prompt_fn is None:
                raise ValueError(f"Task '{cls.name}' must define prompt_fn")
            _registry[cls.name] = cls

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get the task's config, with defaults applied."""
        return cls.default_config or {}


def get_task(name: str) -> Type[BaseTask]:
    """Get a registered task by name."""
    if name not in _registry:
        available = list(_registry.keys())
        raise KeyError(f"Task '{name}' not registered. Available: {available}")
    return _registry[name]


def list_tasks(category: Optional[str] = None) -> list[str]:
    """List all registered task names, optionally filtered by category."""
    if category is None:
        return list(_registry.keys())
    return [name for name, task in _registry.items() if task.category == category]


def get_all_tasks() -> Dict[str, Type[BaseTask]]:
    """Get all registered tasks as a dictionary."""
    return _registry.copy()
