"""
Shared models for classification output.

These models define the structure of extraction results across all classification tasks.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""

    source: Literal["organic", "synthetic"] = "organic"
    processed_at: datetime = Field(default_factory=datetime.now)
    tasks_applied: List[str] = Field(default_factory=list)


class ExtractedText(BaseModel):
    """
    A single text with all its extracted task data.

    This is the text-centric view of extraction results.
    """

    text_id: str
    original_text: str
    results: Dict[str, Any] = Field(default_factory=dict)
    metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)


class ExtractionBatch(BaseModel):
    """
    A batch of extracted texts.

    This is the primary output format for the extraction process.
    """

    batch_id: str
    texts: List[ExtractedText] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    def save(self, filepath: str) -> None:
        """Save batch to JSON file."""
        from pathlib import Path

        Path(filepath).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str) -> "ExtractionBatch":
        """Load batch from JSON file."""
        from pathlib import Path

        return cls.model_validate_json(Path(filepath).read_text())


# --- Excerpt Bank Models (label-centric view) ---


class ExcerptReference(BaseModel):
    """
    A single excerpt with a back-reference to its source.

    Used in the label-centric excerpt bank.
    """

    excerpt: str
    source_text_id: str
    task: str
    label: str
    reasoning: str = ""
    additional_fields: Dict[str, Any] = Field(default_factory=dict)


class ExcerptBank(BaseModel):
    """
    Label-centric index of excerpts.

    Maps labels to lists of excerpts that exhibit that label.
    Built as a derived artifact from ExtractionBatch results.
    """

    label_index: Dict[str, List[ExcerptReference]] = Field(default_factory=dict)
    built_at: datetime = Field(default_factory=datetime.now)
    source_batch_ids: List[str] = Field(default_factory=list)

    def add_excerpt(self, label: str, excerpt_ref: ExcerptReference) -> None:
        """Add an excerpt reference under a label."""
        if label not in self.label_index:
            self.label_index[label] = []
        self.label_index[label].append(excerpt_ref)

    def get_excerpts(self, label: str) -> List[ExcerptReference]:
        """Get all excerpts for a given label."""
        return self.label_index.get(label, [])

    def list_labels(self) -> List[str]:
        """List all labels in the bank."""
        return list(self.label_index.keys())

    def count_by_label(self) -> Dict[str, int]:
        """Get counts of excerpts per label."""
        return {label: len(excerpts) for label, excerpts in self.label_index.items()}

    def save(self, filepath: str) -> None:
        """Save bank to JSON file."""
        from pathlib import Path

        Path(filepath).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str) -> "ExcerptBank":
        """Load bank from JSON file."""
        from pathlib import Path

        return cls.model_validate_json(Path(filepath).read_text())
