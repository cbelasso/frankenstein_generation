"""
Models for validation results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LabelMatch(BaseModel):
    """Result of matching expected vs detected labels."""

    expected: List[str]
    detected: List[str]
    matched: List[str]
    missed: List[str]
    extra: List[str]
    match_ratio: float


class ValidatedText(BaseModel):
    """A synthetic text with validation results."""

    text_id: str
    text: str
    source_labels: List[str]
    target_labels: List[str]
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    label_match: Optional[LabelMatch] = None
    is_valid: bool = False


class ValidationBatch(BaseModel):
    """A batch of validated synthetic texts."""

    batch_id: str
    source_synthetic_batch_id: str
    texts: List[ValidatedText] = Field(default_factory=list)
    tasks_used: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    # Aggregate metrics
    total_texts: int = 0
    valid_count: int = 0
    validation_rate: float = 0.0
    avg_match_ratio: float = 0.0

    def save(self, filepath: str) -> None:
        """Save batch to JSON file."""
        from pathlib import Path

        Path(filepath).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, filepath: str) -> "ValidationBatch":
        """Load batch from JSON file."""
        from pathlib import Path

        return cls.model_validate_json(Path(filepath).read_text())

    def summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "total_texts": self.total_texts,
            "valid_count": self.valid_count,
            "validation_rate": f"{self.validation_rate:.1%}",
            "avg_match_ratio": f"{self.avg_match_ratio:.1%}",
            "tasks_used": self.tasks_used,
        }
