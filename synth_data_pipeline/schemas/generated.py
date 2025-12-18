"""Generated output schemas."""

from typing import List

from pydantic import BaseModel

from .base import GeneratedOutput


class GeneratedComment(GeneratedOutput):
    """Generated comment with alert labels."""
    
    comment: str
    source_excerpts: List[str]
    target_labels: List[str]
    
    def get_target_labels(self) -> List[str]:
        return self.target_labels


class GeneratedNonAlertComment(GeneratedOutput):
    """Generated comment with non-alert classification."""
    
    comment: str
    target_classification: str
    
    def get_target_labels(self) -> List[str]:
        return [self.target_classification]


class StyleTransferOutput(BaseModel):
    """Output from style transfer."""
    
    rewritten_comment: str
