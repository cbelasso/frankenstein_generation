"""
Common input models for tasks.

These define the shapes of input data that tasks can accept.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    """Simple text input for classification tasks."""

    text: str
    text_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TextPairInput(BaseModel):
    """Pair of texts for tasks like style transfer."""

    source_text: str
    reference_text: str
    text_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExcerptSetInput(BaseModel):
    """Set of excerpts for composition tasks."""

    excerpts: List[str]
    target_labels: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
