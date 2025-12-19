"""
Composition task output model.
"""

from pydantic import BaseModel


class CompositionOutput(BaseModel):
    """Output from the composition task."""

    composed_text: str
    coherence_notes: str = ""
