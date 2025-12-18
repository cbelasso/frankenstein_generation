"""Base prompt utilities."""

from typing import Any, Callable, Dict


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with the given kwargs."""
    return template.format(**kwargs)


class PromptTemplate:
    """A reusable prompt template."""
    
    def __init__(self, template: str, name: str = "", description: str = ""):
        self.template = template
        self.name = name
        self.description = description
    
    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)
    
    def __call__(self, **kwargs) -> str:
        return self.format(**kwargs)
