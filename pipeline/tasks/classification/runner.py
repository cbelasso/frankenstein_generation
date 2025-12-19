"""
Classification runner.

Orchestrates running classification tasks and aggregating results.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from pipeline.io import TextRecord
from pipeline.tasks import get_task, run_task
from pipeline.tasks.models import TextInput

from .models import (
    ExcerptBank,
    ExcerptReference,
    ExtractedText,
    ExtractionBatch,
    ExtractionMetadata,
)


def run_classification(
    records: List[Union[TextRecord, str]],
    tasks: List[str],
    processor: Any,
    batch_id: Optional[str] = None,
    output_path: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    batch_size: int = 25,
    source: str = "organic",
) -> ExtractionBatch:
    """
    Run classification tasks on a list of texts and aggregate results.

    Args:
        records: List of TextRecord objects or raw strings
        tasks: List of task names to run (e.g., ["alerts", "recommendations"])
        processor: Processor instance (must implement ProcessorProtocol)
        batch_id: Optional batch ID. Auto-generated if not provided.
        output_path: Optional path to save the ExtractionBatch JSON
        config_overrides: Optional config overrides for all tasks
        batch_size: Batch size for processing
        source: Source type for metadata ("organic" or "synthetic")

    Returns:
        ExtractionBatch containing all extraction results
    """
    # Normalize records
    normalized = _normalize_records(records)

    # Initialize results structure
    results_by_id: Dict[str, Dict[str, Any]] = {r.text_id: {} for r in normalized}

    # Run each task
    for task_name in tasks:
        task_cls = get_task(task_name)

        # Build inputs
        inputs = [TextInput(text=r.text, text_id=r.text_id) for r in normalized]

        # Run task
        task_results = run_task(
            task=task_cls,
            inputs=inputs,
            processor=processor,
            config_overrides=config_overrides,
            batch_size=batch_size,
        )

        # Aggregate results by text_id
        for record, result in zip(normalized, task_results):
            if result is not None:
                results_by_id[record.text_id][task_name] = result.model_dump()

    # Build ExtractionBatch
    batch = ExtractionBatch(
        batch_id=batch_id or f"batch_{uuid.uuid4().hex[:8]}",
        texts=[
            ExtractedText(
                text_id=r.text_id,
                original_text=r.text,
                results=results_by_id[r.text_id],
                metadata=ExtractionMetadata(
                    source=source,
                    processed_at=datetime.now(),
                    tasks_applied=tasks,
                ),
            )
            for r in normalized
        ],
        created_at=datetime.now(),
    )

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        batch.save(output_path)

    return batch


def _normalize_records(records: List[Union[TextRecord, str]]) -> List[TextRecord]:
    """Normalize inputs to TextRecord objects."""
    normalized = []
    auto_id = 0

    for r in records:
        if isinstance(r, str):
            auto_id += 1
            normalized.append(TextRecord(text_id=f"text_{auto_id:04d}", text=r))
        elif isinstance(r, TextRecord):
            normalized.append(r)
        else:
            raise TypeError(f"Expected TextRecord or str, got {type(r)}")

    return normalized


def build_excerpt_bank(
    batch: Union[ExtractionBatch, str],
    output_path: Optional[str] = None,
    task_label_extractors: Optional[Dict[str, callable]] = None,
) -> ExcerptBank:
    """
    Build a label-centric ExcerptBank from an ExtractionBatch.

    Args:
        batch: ExtractionBatch object or path to JSON file
        output_path: Optional path to save the ExcerptBank JSON
        task_label_extractors: Optional dict mapping task names to custom
            extractor functions. If not provided, uses default extractors.

    Returns:
        ExcerptBank with excerpts indexed by label
    """
    # Load batch if path
    if isinstance(batch, str):
        batch = ExtractionBatch.load(batch)

    # Default extractors for known tasks
    extractors = {
        "alerts": _extract_alerts,
        "recommendations": _extract_recommendations,
    }

    # Override with custom extractors
    if task_label_extractors:
        extractors.update(task_label_extractors)

    # Build bank
    bank = ExcerptBank(
        source_batch_ids=[batch.batch_id],
        built_at=datetime.now(),
    )

    for text in batch.texts:
        for task_name, result in text.results.items():
            if task_name in extractors:
                excerpts = extractors[task_name](
                    result=result,
                    source_text_id=text.text_id,
                    task=task_name,
                )
                for label, excerpt_ref in excerpts:
                    bank.add_excerpt(label, excerpt_ref)

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        bank.save(output_path)

    return bank


def _extract_alerts(
    result: Dict[str, Any],
    source_text_id: str,
    task: str,
) -> List[tuple]:
    """Extract labeled excerpts from alerts results."""
    excerpts = []

    if not result.get("has_alerts", False):
        return excerpts

    for alert in result.get("alerts", []):
        label = alert.get("alert_type")
        if label and alert.get("excerpt"):
            excerpt_ref = ExcerptReference(
                excerpt=alert["excerpt"],
                source_text_id=source_text_id,
                task=task,
                label=label,
                reasoning=alert.get("reasoning", ""),
                additional_fields={
                    "severity": alert.get("severity"),
                },
            )
            excerpts.append((label, excerpt_ref))

    return excerpts


def _extract_recommendations(
    result: Dict[str, Any],
    source_text_id: str,
    task: str,
) -> List[tuple]:
    """Extract labeled excerpts from recommendations results."""
    excerpts = []

    if not result.get("has_recommendations", False):
        return excerpts

    for rec in result.get("recommendations", []):
        label = rec.get("qualifier")
        if label and rec.get("excerpt"):
            excerpt_ref = ExcerptReference(
                excerpt=rec["excerpt"],
                source_text_id=source_text_id,
                task=task,
                label=label,
                reasoning=rec.get("reasoning", ""),
                additional_fields={
                    "paraphrased": rec.get("paraphrased_recommendation"),
                },
            )
            excerpts.append((label, excerpt_ref))

    return excerpts
