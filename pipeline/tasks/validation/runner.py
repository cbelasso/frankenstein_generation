"""
Validation runner.

Re-runs classification on synthetic texts to verify intended labels manifest.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import uuid

from pipeline.tasks import get_task, run_task
from pipeline.tasks.generation.models import SyntheticBatch
from pipeline.tasks.models import TextInput

from .models import LabelMatch, ValidatedText, ValidationBatch


def run_validation(
    synthetic_batch: Union[SyntheticBatch, str],
    processor: Any,
    tasks: List[str] = None,
    batch_id: Optional[str] = None,
    output_path: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    batch_size: int = 25,
    match_threshold: float = 0.5,
) -> ValidationBatch:
    """
    Validate synthetic texts by re-running classification.

    Args:
        synthetic_batch: SyntheticBatch object or path to JSON file
        processor: Processor instance
        tasks: List of classification tasks to run (default: ["alerts", "recommendations"])
        batch_id: Optional batch ID
        output_path: Optional path to save ValidationBatch JSON
        config_overrides: Optional config overrides
        batch_size: Batch size for processing
        match_threshold: Minimum match ratio to consider valid (0.0-1.0)

    Returns:
        ValidationBatch with validation results
    """
    # Load batch if path
    if isinstance(synthetic_batch, str):
        synthetic_batch = SyntheticBatch.load(synthetic_batch)

    tasks = tasks or ["alerts", "recommendations"]

    # Prepare inputs
    texts = [t.text for t in synthetic_batch.texts]
    text_ids = [t.text_id for t in synthetic_batch.texts]

    # Run each classification task
    all_results: Dict[str, List[Any]] = {}
    for task_name in tasks:
        task_cls = get_task(task_name)
        inputs = [TextInput(text=t, text_id=tid) for t, tid in zip(texts, text_ids)]

        results = run_task(
            task=task_cls,
            inputs=inputs,
            processor=processor,
            config_overrides=config_overrides,
            batch_size=batch_size,
        )
        all_results[task_name] = results

    # Build validated texts
    validated_texts = []
    for i, synth_text in enumerate(synthetic_batch.texts):
        # Collect results for this text
        validation_results = {}
        detected_labels: Set[str] = set()

        for task_name in tasks:
            result = all_results[task_name][i]
            if result is not None:
                result_dict = result.model_dump()
                validation_results[task_name] = result_dict

                # Extract detected labels
                detected_labels.update(_extract_labels_from_result(task_name, result_dict))

        # Calculate label match
        expected = set(synth_text.target_labels)
        matched = expected & detected_labels
        missed = expected - detected_labels
        extra = detected_labels - expected

        match_ratio = len(matched) / len(expected) if expected else 1.0

        label_match = LabelMatch(
            expected=list(expected),
            detected=list(detected_labels),
            matched=list(matched),
            missed=list(missed),
            extra=list(extra),
            match_ratio=match_ratio,
        )

        validated_texts.append(
            ValidatedText(
                text_id=synth_text.text_id,
                text=synth_text.text,
                source_labels=synth_text.source_labels,
                target_labels=synth_text.target_labels,
                validation_results=validation_results,
                label_match=label_match,
                is_valid=match_ratio >= match_threshold,
            )
        )

    # Calculate aggregate metrics
    total = len(validated_texts)
    valid_count = sum(1 for t in validated_texts if t.is_valid)
    avg_match = (
        sum(t.label_match.match_ratio for t in validated_texts) / total if total > 0 else 0.0
    )

    validation_batch = ValidationBatch(
        batch_id=batch_id or f"validation_{uuid.uuid4().hex[:8]}",
        source_synthetic_batch_id=synthetic_batch.batch_id,
        texts=validated_texts,
        tasks_used=tasks,
        created_at=datetime.now(),
        total_texts=total,
        valid_count=valid_count,
        validation_rate=valid_count / total if total > 0 else 0.0,
        avg_match_ratio=avg_match,
    )

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        validation_batch.save(output_path)

    return validation_batch


def _extract_labels_from_result(task_name: str, result: Dict) -> Set[str]:
    """Extract labels from a classification result."""
    labels = set()

    if task_name == "alerts":
        if result.get("has_alerts"):
            for alert in result.get("alerts", []):
                if alert.get("alert_type"):
                    labels.add(alert["alert_type"])

    elif task_name == "recommendations":
        if result.get("has_recommendations"):
            for rec in result.get("recommendations", []):
                if rec.get("qualifier"):
                    labels.add(rec["qualifier"])

    return labels


def get_valid_texts(validation_batch: ValidationBatch) -> List[ValidatedText]:
    """Get only valid texts from a validation batch."""
    return [t for t in validation_batch.texts if t.is_valid]


def get_invalid_texts(validation_batch: ValidationBatch) -> List[ValidatedText]:
    """Get only invalid texts from a validation batch."""
    return [t for t in validation_batch.texts if not t.is_valid]


def filter_by_match_ratio(
    validation_batch: ValidationBatch, min_ratio: float
) -> List[ValidatedText]:
    """Filter texts by minimum match ratio."""
    return [
        t
        for t in validation_batch.texts
        if t.label_match and t.label_match.match_ratio >= min_ratio
    ]
