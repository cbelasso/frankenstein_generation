"""
Generation tasks.

Tasks for generating synthetic data from excerpt banks.
"""

from . import composition
from .models import (
    CompositionOutput,
    ExcerptSet,
    SyntheticBatch,
    SyntheticText,
)
from .runner import run_composition
from .sampling import (
    sample_by_label_combination,
    sample_random_combinations,
    sample_underrepresented_combinations,
)

__all__ = [
    # Submodules
    "composition",
    # Models
    "ExcerptSet",
    "CompositionOutput",
    "SyntheticText",
    "SyntheticBatch",
    # Sampling
    "sample_by_label_combination",
    "sample_random_combinations",
    "sample_underrepresented_combinations",
    # Runner
    "run_composition",
]
