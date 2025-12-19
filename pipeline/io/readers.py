"""
Input data readers.

Utilities for loading text data from various sources.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class TextRecord:
    """A text record with its ID and original text."""

    text_id: str
    text: str


def load_texts_from_csv(
    filepath: Union[str, Path],
    text_column: str,
    id_column: Optional[str] = None,
    id_prefix: str = "text",
    encoding: str = "utf-8",
    skip_empty: bool = True,
) -> List[TextRecord]:
    """
    Load texts from a CSV file.

    Args:
        filepath: Path to the CSV file
        text_column: Name of the column containing the text
        id_column: Optional column name for text IDs. If None, IDs are auto-generated.
        id_prefix: Prefix for auto-generated IDs (e.g., "text" -> "text_001")
        encoding: File encoding (default: utf-8)
        skip_empty: Whether to skip rows with empty/whitespace-only text

    Returns:
        List of TextRecord objects

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the specified column doesn't exist
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    records = []
    auto_id_counter = 0

    with open(filepath, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)

        # Validate columns exist
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no header row")

        if text_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{text_column}' not found. Available: {reader.fieldnames}"
            )

        if id_column and id_column not in reader.fieldnames:
            raise ValueError(
                f"ID column '{id_column}' not found. Available: {reader.fieldnames}"
            )

        for row in reader:
            text = row[text_column]

            # Skip empty texts if requested
            if skip_empty and (not text or not text.strip()):
                continue

            # Get or generate ID
            if id_column:
                text_id = str(row[id_column])
            else:
                auto_id_counter += 1
                text_id = f"{id_prefix}_{auto_id_counter:04d}"

            records.append(TextRecord(text_id=text_id, text=text))

    return records


def load_texts_as_strings(
    filepath: Union[str, Path],
    text_column: str,
    encoding: str = "utf-8",
    skip_empty: bool = True,
) -> List[str]:
    """
    Load texts from a CSV file as plain strings.

    Convenience function when you don't need IDs.

    Args:
        filepath: Path to the CSV file
        text_column: Name of the column containing the text
        encoding: File encoding (default: utf-8)
        skip_empty: Whether to skip rows with empty/whitespace-only text

    Returns:
        List of text strings
    """
    records = load_texts_from_csv(
        filepath=filepath,
        text_column=text_column,
        encoding=encoding,
        skip_empty=skip_empty,
    )
    return [r.text for r in records]
