"""
Facet registry for classification capabilities.

Facets auto-register when their class is defined by inheriting from BaseFacet.
"""

from typing import Callable, Dict, Type

from pydantic import BaseModel

_registry: Dict[str, Type["BaseFacet"]] = {}


class BaseFacet:
    """
    Base class for all classification facets.

    Subclasses must define:
        - name: str - unique identifier for the facet
        - output_model: Type[BaseModel] - Pydantic model for the output schema
        - prompt_fn: Callable[[str], str] - function that generates the prompt

    Registration happens automatically via __init_subclass__.
    """

    name: str | None = None
    output_model: Type[BaseModel] | None = None
    prompt_fn: Callable[[str], str] | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            if cls.output_model is None:
                raise ValueError(f"Facet '{cls.name}' must define output_model")
            if cls.prompt_fn is None:
                raise ValueError(f"Facet '{cls.name}' must define prompt_fn")
            _registry[cls.name] = cls


def get_facet(name: str) -> Type[BaseFacet]:
    """Get a registered facet by name."""
    if name not in _registry:
        available = list(_registry.keys())
        raise KeyError(f"Facet '{name}' not registered. Available: {available}")
    return _registry[name]


def list_facets() -> list[str]:
    """List all registered facet names."""
    return list(_registry.keys())


def get_all_facets() -> Dict[str, Type[BaseFacet]]:
    """Get all registered facets as a dictionary."""
    return _registry.copy()
