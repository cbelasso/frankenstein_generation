#!/usr/bin/env python3
"""
Main script for synthetic data generation pipeline.

Usage:
    python main.py --input data.csv --column comment_text --gpu 0 1 2 3
    python main.py --input data.xlsx --column feedback --gpu 0 --n-samples 5
    python main.py --input data.json --column text --gpu 0 1 --target-labels bullying discrimination
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from llm_parallelization.new_processor import NewProcessor
import pandas as pd

from synth_data_pipeline.data_store import DataStore
from synth_data_pipeline.extraction import AlertsExtractor
from synth_data_pipeline.generation import Generator
from synth_data_pipeline.schemas import AlertsOutput, GeneratedComment, GeneratedNonAlertComment
from synth_data_pipeline.validation import Validator, print_validation_result


def load_data(filepath: str, column: str) -> List[str]:
    """Load text data from CSV, Excel, or JSON."""
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available: {list(df.columns)}")

    texts = df[column].dropna().astype(str).tolist()
    print(f"✓ Loaded {len(texts)} texts from '{column}' column")
    return texts


def run_extraction(
    processor: NewProcessor,
    texts: List[str],
    batch_size: int = 25,
) -> tuple[DataStore, List[AlertsOutput]]:
    """Run extraction on texts and populate data store."""
    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTION")
    print("=" * 60)

    extractor = AlertsExtractor(processor)
    extractions = extractor.extract(texts, batch_size=batch_size)

    store = DataStore()
    store.load_from_extraction_results(texts, extractions)

    print(f"\n✓ Extracted {len(store.excerpts_df)} alert excerpts from {len(texts)} comments")
    print(f"✓ Labels found: {store.get_available_labels()}")
    print(f"✓ Label counts: {store.get_label_counts()}")

    return store, extractions


def run_generation(
    processor: NewProcessor,
    store: DataStore,
    target_labels: List[str],
    n_samples: int = 5,
    n_candidates: int = 5,
) -> List[GeneratedComment]:
    """Generate synthetic alert comments for target labels."""
    print("\n" + "=" * 60)
    print("STEP 2a: ALERT GENERATION")
    print("=" * 60)

    print(f"\nTarget labels: {target_labels}")
    print(f"Generating {n_samples} samples...")

    generator = Generator(processor, store)
    generated = generator.generate_from_labels(
        target_labels=target_labels,
        n_samples=n_samples,
        n_candidates_per_label=n_candidates,
    )

    print(f"\n✓ Generated {len(generated)} alert comments")
    for i, g in enumerate(generated, 1):
        print(f"\n[{i}] {g.comment[:100]}...")

    return generated


def run_non_alert_generation(
    processor: NewProcessor,
    store: DataStore,
    target_classification: str,
    n_samples: int = 5,
    n_references: int = 5,
) -> List[GeneratedNonAlertComment]:
    """Generate synthetic non-alert comments for target classification."""
    print("\n" + "=" * 60)
    print("STEP 2b: NON-ALERT GENERATION")
    print("=" * 60)

    print(f"\nTarget classification: {target_classification}")
    print(f"Generating {n_samples} samples...")

    generator = Generator(processor, store)
    generated = generator.generate_non_alerts(
        target_classification=target_classification,
        n_samples=n_samples,
        n_references=n_references,
    )

    print(f"\n✓ Generated {len(generated)} non-alert comments")
    for i, g in enumerate(generated, 1):
        print(f"\n[{i}] {g.comment[:100]}...")

    return generated


def run_validation(
    processor: NewProcessor,
    generated: List[GeneratedComment],
) -> None:
    """Validate generated comments by re-extracting and comparing."""
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATION")
    print("=" * 60)

    extractor = AlertsExtractor(processor)
    validator = Validator(extractor, match_threshold=0.5)

    results = validator.validate_batch(generated)

    for result in results:
        print_validation_result(result)

    summary = validator.summary(results)
    print("\n" + "-" * 40)
    print("SUMMARY:")
    print(f"  Pass rate: {summary['pass_rate']:.1%} ({summary['passed']}/{summary['total']})")
    print(f"  Avg Precision: {summary['avg_precision']:.2f}")
    print(f"  Avg Recall: {summary['avg_recall']:.2f}")
    print(f"  Avg F1: {summary['avg_f1']:.2f}")

    if summary.get("commonly_missed_labels"):
        print(f"  Commonly missed: {summary['commonly_missed_labels']}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic data generation pipeline")

    # Data loading
    parser.add_argument("--input", "-i", required=True, help="Input file (CSV, Excel, or JSON)")
    parser.add_argument(
        "--column", "-c", required=True, help="Column name containing text data"
    )

    # GPU configuration
    parser.add_argument("--gpu", "-g", nargs="+", type=int, default=[0], help="GPU IDs to use")
    parser.add_argument(
        "--model",
        "-m",
        default="casperhansen/mistral-nemo-instruct-2407-awq",
        help="Model path",
    )

    # Pipeline options
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for processing")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to process (for testing)"
    )

    # Generation options
    parser.add_argument(
        "--generate",
        choices=["alerts", "non-alerts", "both"],
        default="alerts",
        help="What type of comments to generate",
    )
    parser.add_argument(
        "--target-labels",
        "-t",
        nargs="+",
        default=None,
        help="Target alert labels for generation",
    )
    parser.add_argument(
        "--target-classification",
        default=None,
        help="Target non-alert classification (e.g., positive_feedback, resource_request)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5, help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--n-candidates", type=int, default=5, help="Candidate excerpts/references per label"
    )

    # Output
    parser.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    parser.add_argument("--skip-generation", action="store_true", help="Only run extraction")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")

    args = parser.parse_args()

    # Load data
    texts = load_data(args.input, args.column)

    if args.max_samples:
        texts = texts[: args.max_samples]
        print(f"✓ Limited to {len(texts)} samples for testing")

    # Initialize processor
    print(f"\n✓ Initializing NewProcessor on GPUs: {args.gpu}")
    print(f"✓ Model: {args.model}")

    with NewProcessor(
        gpu_list=args.gpu,
        llm=args.model,
        multiplicity=1,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
    ) as processor:
        # Step 1: Extraction
        store, extractions = run_extraction(processor, texts, args.batch_size)

        # Save extraction results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        store.save(output_dir / "data_store")
        print(f"\n✓ Saved data store to {output_dir / 'data_store'}")

        if args.skip_generation:
            print("\n✓ Skipping generation (--skip-generation)")
            return

        all_generated_alerts = []
        all_generated_non_alerts = []

        # Generate alerts
        if args.generate in ["alerts", "both"]:
            # Determine target labels
            if args.target_labels:
                target_labels = args.target_labels
            else:
                available = store.get_available_labels()
                if len(available) >= 2:
                    target_labels = available[:2]
                elif len(available) == 1:
                    target_labels = available
                else:
                    target_labels = []
                    print("\n⚠ No alert labels found in data, skipping alert generation")

            if target_labels:
                # Verify target labels exist
                available_labels = set(store.get_available_labels())
                missing = [l for l in target_labels if l not in available_labels]
                if missing:
                    print(f"\n⚠ Warning: Labels not in data: {missing}")
                    print(f"  Available labels: {list(available_labels)}")
                    target_labels = [l for l in target_labels if l in available_labels]

                if target_labels:
                    generated_alerts = run_generation(
                        processor,
                        store,
                        target_labels,
                        n_samples=args.n_samples,
                        n_candidates=args.n_candidates,
                    )
                    all_generated_alerts.extend(generated_alerts)

        # Generate non-alerts
        if args.generate in ["non-alerts", "both"]:
            # Determine target classification
            if args.target_classification:
                target_classifications = [args.target_classification]
            else:
                available = store.get_available_non_alert_classes()
                if available:
                    target_classifications = available[:2]  # Default to first 2
                else:
                    target_classifications = []
                    print(
                        "\n⚠ No non-alert classifications found in data, skipping non-alert generation"
                    )

            for classification in target_classifications:
                available_classes = set(store.get_available_non_alert_classes())
                if classification not in available_classes:
                    print(f"\n⚠ Warning: Classification '{classification}' not in data")
                    print(f"  Available: {list(available_classes)}")
                    continue

                generated_non_alerts = run_non_alert_generation(
                    processor,
                    store,
                    classification,
                    n_samples=args.n_samples,
                    n_references=args.n_candidates,
                )
                all_generated_non_alerts.extend(generated_non_alerts)

        # Save generated comments
        if all_generated_alerts:
            alerts_data = [g.model_dump() for g in all_generated_alerts if g is not None]
            with open(output_dir / "generated_alerts.json", "w") as f:
                json.dump(alerts_data, f, indent=2)
            print(
                f"\n✓ Saved {len(alerts_data)} generated alerts to {output_dir / 'generated_alerts.json'}"
            )

        if all_generated_non_alerts:
            non_alerts_data = [
                g.model_dump() for g in all_generated_non_alerts if g is not None
            ]
            with open(output_dir / "generated_non_alerts.json", "w") as f:
                json.dump(non_alerts_data, f, indent=2)
            print(
                f"\n✓ Saved {len(non_alerts_data)} generated non-alerts to {output_dir / 'generated_non_alerts.json'}"
            )

        # Validation (only for alerts)
        if not args.skip_validation and all_generated_alerts:
            run_validation(processor, all_generated_alerts)
        elif args.skip_validation:
            print("\n✓ Skipping validation")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
