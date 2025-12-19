"""
IO utilities for the pipeline.

Provides utilities for reading input data and writing output files.
"""

from .readers import TextRecord, load_texts_from_csv

__all__ = ["load_texts_from_csv", "TextRecord"]
