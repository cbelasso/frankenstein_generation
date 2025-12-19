"""
Sampling strategies for selecting excerpts from the bank.
"""

import random
from typing import List, Optional, Set, Tuple

from pipeline.tasks.classification.models import ExcerptBank, ExcerptReference

from .models import ExcerptSet


def sample_by_label_combination(
    bank: ExcerptBank,
    label_combinations: List[List[str]],
    n_per_combination: int = 10,
    max_excerpts_per_sample: int = 3,
    avoid_same_source: bool = True,
    seed: Optional[int] = None,
) -> List[ExcerptSet]:
    """
    Sample excerpt sets targeting specific label combinations.

    Args:
        bank: ExcerptBank to sample from
        label_combinations: List of label lists to target
            e.g., [["discrimination", "negative"], ["fraud", "add_or_increase"]]
        n_per_combination: Number of samples to generate per combination
        max_excerpts_per_sample: Maximum excerpts to include per sample
        avoid_same_source: If True, avoid using excerpts from the same source text
        seed: Random seed for reproducibility

    Returns:
        List of ExcerptSet objects ready for composition
    """
    if seed is not None:
        random.seed(seed)

    excerpt_sets = []

    for labels in label_combinations:
        # Check all labels exist in bank
        available_labels = set(bank.list_labels())
        missing = set(labels) - available_labels
        if missing:
            print(f"Warning: Labels not in bank: {missing}, skipping combination {labels}")
            continue

        for _ in range(n_per_combination):
            excerpt_set = _sample_one_combination(
                bank=bank,
                labels=labels,
                max_excerpts=max_excerpts_per_sample,
                avoid_same_source=avoid_same_source,
            )
            if excerpt_set:
                excerpt_sets.append(excerpt_set)

    return excerpt_sets


def sample_random_combinations(
    bank: ExcerptBank,
    n_samples: int = 50,
    min_labels: int = 1,
    max_labels: int = 3,
    max_excerpts_per_sample: int = 3,
    avoid_same_source: bool = True,
    seed: Optional[int] = None,
) -> List[ExcerptSet]:
    """
    Sample random label combinations from the bank.

    Useful for exploring what combinations naturally emerge.

    Args:
        bank: ExcerptBank to sample from
        n_samples: Total number of samples to generate
        min_labels: Minimum number of distinct labels per sample
        max_labels: Maximum number of distinct labels per sample
        max_excerpts_per_sample: Maximum excerpts per sample
        avoid_same_source: If True, avoid using excerpts from the same source text
        seed: Random seed for reproducibility

    Returns:
        List of ExcerptSet objects
    """
    if seed is not None:
        random.seed(seed)

    available_labels = bank.list_labels()
    if not available_labels:
        return []

    excerpt_sets = []

    for _ in range(n_samples):
        # Pick random number of labels
        n_labels = random.randint(min_labels, min(max_labels, len(available_labels)))
        labels = random.sample(available_labels, n_labels)

        excerpt_set = _sample_one_combination(
            bank=bank,
            labels=labels,
            max_excerpts=max_excerpts_per_sample,
            avoid_same_source=avoid_same_source,
        )
        if excerpt_set:
            excerpt_sets.append(excerpt_set)

    return excerpt_sets


def sample_underrepresented_combinations(
    bank: ExcerptBank,
    label_counts: dict,
    n_samples: int = 50,
    threshold_percentile: float = 25.0,
    max_excerpts_per_sample: int = 3,
    avoid_same_source: bool = True,
    seed: Optional[int] = None,
) -> List[ExcerptSet]:
    """
    Sample combinations that prioritize underrepresented labels.

    Args:
        bank: ExcerptBank to sample from
        label_counts: Dict of label -> count from your original dataset
        n_samples: Number of samples to generate
        threshold_percentile: Labels below this percentile are considered underrepresented
        max_excerpts_per_sample: Maximum excerpts per sample
        avoid_same_source: Avoid excerpts from same source text
        seed: Random seed

    Returns:
        List of ExcerptSet objects targeting rare labels
    """
    if seed is not None:
        random.seed(seed)

    if not label_counts:
        return sample_random_combinations(
            bank, n_samples, max_excerpts_per_sample=max_excerpts_per_sample, seed=seed
        )

    # Find threshold
    counts = list(label_counts.values())
    counts.sort()
    threshold_idx = int(len(counts) * threshold_percentile / 100)
    threshold = counts[threshold_idx] if threshold_idx < len(counts) else counts[-1]

    # Get underrepresented labels that exist in bank
    available_labels = set(bank.list_labels())
    rare_labels = [
        label
        for label, count in label_counts.items()
        if count <= threshold and label in available_labels
    ]

    if not rare_labels:
        print("No underrepresented labels found in bank, using random sampling")
        return sample_random_combinations(
            bank, n_samples, max_excerpts_per_sample=max_excerpts_per_sample, seed=seed
        )

    excerpt_sets = []

    for _ in range(n_samples):
        # Always include at least one rare label
        n_rare = random.randint(1, min(2, len(rare_labels)))
        labels = random.sample(rare_labels, n_rare)

        # Optionally add a common label for variety
        other_labels = list(available_labels - set(labels))
        if other_labels and random.random() > 0.5:
            labels.append(random.choice(other_labels))

        excerpt_set = _sample_one_combination(
            bank=bank,
            labels=labels,
            max_excerpts=max_excerpts_per_sample,
            avoid_same_source=avoid_same_source,
        )
        if excerpt_set:
            excerpt_sets.append(excerpt_set)

    return excerpt_sets


def _sample_one_combination(
    bank: ExcerptBank,
    labels: List[str],
    max_excerpts: int,
    avoid_same_source: bool,
) -> Optional[ExcerptSet]:
    """Sample one excerpt set for a given label combination."""
    excerpts = []
    source_labels = []
    source_text_ids = []
    used_sources: Set[str] = set()

    for label in labels:
        available = bank.get_excerpts(label)

        if avoid_same_source:
            available = [e for e in available if e.source_text_id not in used_sources]

        if not available:
            continue

        # Sample one excerpt for this label
        excerpt_ref = random.choice(available)
        excerpts.append(excerpt_ref.excerpt)
        source_labels.append(label)
        source_text_ids.append(excerpt_ref.source_text_id)
        used_sources.add(excerpt_ref.source_text_id)

        if len(excerpts) >= max_excerpts:
            break

    if not excerpts:
        return None

    return ExcerptSet(
        excerpts=excerpts,
        source_labels=source_labels,
        source_text_ids=source_text_ids,
        target_labels=labels,
    )
