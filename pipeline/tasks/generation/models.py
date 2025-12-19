"""
Models for synthetic data generation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ExcerptSet(BaseModel):
    """A set of excerpts to be composed into a synthetic text."""

    excerpts: List[str]
    source_labels: List[str]
    source_text_ids: List[str] = Field(default_factory=list)
    target_labels: List[str] = Field(default_factory=list)


class CompositionOutput(BaseModel):
    """Output from the composition task."""

    composed_text: str
    coherence_notes: str = ""


class SyntheticText(BaseModel):
    """A synthetic text with provenance metadata."""

    text_id: str
    text: str
    source_excerpts: List[str]
    source_labels: List[str]
    source_text_ids: List[str]
    target_labels: List[str]
    coherence_notes: str = ""
    created_at: datetime = Field(default_factory=datetime.now)


class SyntheticBatch(BaseModel):
    """A batch of synthetic texts."""

    batch_id: str
    texts: List[SyntheticText] = Field(default_factory=list)
    source_excerpt_bank: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    def save(self, filepath: str) -> None:
        """Save batch to JSON file."""
        from pathlib import Path

        Path(filepath).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str) -> "SyntheticBatch":
        """Load batch from JSON file."""
        from pathlib import Path

        return cls.model_validate_json(Path(filepath).read_text())

    def to_texts_for_validation(self) -> List[Dict[str, str]]:
        """Convert to format suitable for running validation."""
        return [{"text_id": t.text_id, "text": t.text} for t in self.texts]
