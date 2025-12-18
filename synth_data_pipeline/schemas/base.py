"""Base schema classes and interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel


class ExtractionOutput(BaseModel, ABC):
    """Base class for all extraction output schemas."""
    
    @abstractmethod
    def get_labels(self) -> Set[str]:
        """Return the set of labels/categories extracted."""
        pass
    
    @abstractmethod
    def get_excerpts(self) -> List[Dict[str, Any]]:
        """Return list of extracted excerpts with metadata."""
        pass
    
    @abstractmethod
    def has_positive_labels(self) -> bool:
        """Return True if any positive labels were found."""
        pass
    
    @abstractmethod
    def get_negative_classification(self) -> Optional[str]:
        """Return the negative/non-match classification if any."""
        pass


class GeneratedOutput(BaseModel, ABC):
    """Base class for all generated output schemas."""
    
    comment: str
    
    @abstractmethod
    def get_target_labels(self) -> List[str]:
        """Return the target labels this was generated for."""
        pass
