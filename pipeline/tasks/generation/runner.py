"""
Generation runner.

Orchestrates synthetic data generation from excerpt banks.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

from pipeline.tasks import get_task, run_task
from pipeline.tasks.classification.models import ExcerptBank

from .models import ExcerptSet, SyntheticBatch, SyntheticText
from .sampling import (
    sample_by_label_combination,
    sample_random_combinations,
    sample_underrepresented_combinations,
)


def run_composition(
    excerpt_bank: Union[ExcerptBank, str],
    processor: Any,
    excerpt_sets: Optional[List[ExcerptSet]] = None,
    target_combinations: Optional[List[List[str]]] = None,
    n_samples: int = 50,
    n_per_combination: int = 10,
    sampling_strategy: str = "random",
    batch_id: Optional[str] = None,
    output_path: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    batch_size: int = 25,
    seed: Optional[int] = None,
    label_counts: Optional[Dict[str, int]] = None,
) -> SyntheticBatch:
    """
    Generate synthetic texts by composing excerpts from an excerpt bank.

    Args:
        excerpt_bank: ExcerptBank object or path to JSON file
        processor: Processor instance (must implement ProcessorProtocol)
        excerpt_sets: Pre-built excerpt sets to use (skips sampling if provided)
        target_combinations: Specific label combinations to target
            e.g., [["discrimination", "negative"], ["fraud"]]
        n_samples: Number of samples for random/underrepresented sampling
        n_per_combination: Samples per combination for targeted sampling
        sampling_strategy: One of "random", "targeted", "underrepresented"
        batch_id: Optional batch ID. Auto-generated if not provided.
        output_path: Optional path to save the SyntheticBatch JSON
        config_overrides: Optional config overrides for composition task
        batch_size: Batch size for processing
        seed: Random seed for reproducibility
        label_counts: Label counts from original data (for underrepresented sampling)

    Returns:
        SyntheticBatch containing generated synthetic texts
    """
    # Load bank if path
    if isinstance(excerpt_bank, str):
        bank_path = excerpt_bank
        excerpt_bank = ExcerptBank.load(excerpt_bank)
    else:
        bank_path = None

    # Get or sample excerpt sets
    if excerpt_sets is not None:
        sets_to_compose = excerpt_sets
    else:
        sets_to_compose = _sample_excerpt_sets(
            bank=excerpt_bank,
            strategy=sampling_strategy,
            target_combinations=target_combinations,
            n_samples=n_samples,
            n_per_combination=n_per_combination,
            seed=seed,
            label_counts=label_counts,
        )

    if not sets_to_compose:
        print("Warning: No excerpt sets to compose")
        return SyntheticBatch(
            batch_id=batch_id or f"synthetic_{uuid.uuid4().hex[:8]}",
            texts=[],
            source_excerpt_bank=bank_path,
        )

    # Run composition task
    # Need to import the task to trigger registration
    from .composition import CompositionTask

    results = run_task(
        task=CompositionTask,
        inputs=sets_to_compose,
        processor=processor,
        config_overrides=config_overrides,
        batch_size=batch_size,
    )

    # Build SyntheticBatch
    synthetic_texts = []
    for i, (excerpt_set, result) in enumerate(zip(sets_to_compose, results)):
        if result is None:
            continue

        synthetic_texts.append(
            SyntheticText(
                text_id=f"synthetic_{i + 1:04d}",
                text=result.composed_text,
                source_excerpts=excerpt_set.excerpts,
                source_labels=excerpt_set.source_labels,
                source_text_ids=excerpt_set.source_text_ids,
                target_labels=excerpt_set.target_labels,
                coherence_notes=result.coherence_notes,
                created_at=datetime.now(),
            )
        )

    batch = SyntheticBatch(
        batch_id=batch_id or f"synthetic_{uuid.uuid4().hex[:8]}",
        texts=synthetic_texts,
        source_excerpt_bank=bank_path,
        created_at=datetime.now(),
    )

    # Save if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        batch.save(output_path)

    return batch


def _sample_excerpt_sets(
    bank: ExcerptBank,
    strategy: str,
    target_combinations: Optional[List[List[str]]],
    n_samples: int,
    n_per_combination: int,
    seed: Optional[int],
    label_counts: Optional[Dict[str, int]],
) -> List[ExcerptSet]:
    """Sample excerpt sets based on strategy."""

    if strategy == "targeted" and target_combinations:
        return sample_by_label_combination(
            bank=bank,
            label_combinations=target_combinations,
            n_per_combination=n_per_combination,
            seed=seed,
        )
    elif strategy == "underrepresented":
        return sample_underrepresented_combinations(
            bank=bank,
            label_counts=label_counts or {},
            n_samples=n_samples,
            seed=seed,
        )
    else:  # default to random
        return sample_random_combinations(
            bank=bank,
            n_samples=n_samples,
            seed=seed,
        )
