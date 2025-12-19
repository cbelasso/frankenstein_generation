from llm_parallelization.new_processor import NEMO, NewProcessor

from pipeline.io import load_texts_from_csv
from pipeline.tasks.classification import build_excerpt_bank, run_classification
from pipeline.tasks.generation import run_composition
from pipeline.tasks.validation import get_invalid_texts, get_valid_texts, run_validation

gpu_list = [3, 4, 5, 6, 7]
gpu_memory_utilization = 0.5
llm = NEMO
max_model_len = 8192
multiplicity = 1

if __name__ == "__main__":
    # Load data
    records = load_texts_from_csv(
        "./datasets/data_gov_golden_dataset_comments_sample_100.csv", text_column="comment"
    )

    with NewProcessor(
        gpu_list=gpu_list,
        llm=llm,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        multiplicity=multiplicity,
    ) as processor:
        # Step 1: Run classification
        print("=" * 50)
        print("Step 1: Classification")
        print("=" * 50)
        batch = run_classification(
            records=records,
            tasks=["alerts", "recommendations"],
            processor=processor,
            output_path="output/batch_001.json",
        )
        print(f"Classified {len(batch.texts)} texts")

        # Step 2: Build excerpt bank
        print("\n" + "=" * 50)
        print("Step 2: Build Excerpt Bank")
        print("=" * 50)
        bank = build_excerpt_bank(batch, output_path="output/excerpt_bank.json")
        print(f"Labels: {bank.list_labels()}")
        print(f"Counts: {bank.count_by_label()}")

        # Step 3: Generate synthetic data
        print("\n" + "=" * 50)
        print("Step 3: Generate Synthetic Data")
        print("=" * 50)
        synthetic = run_composition(
            excerpt_bank=bank,
            processor=processor,
            n_samples=50,
            sampling_strategy="random",
            seed=42,
            output_path="output/synthetic_batch_001.json",
        )
        print(f"Generated {len(synthetic.texts)} synthetic texts")

        # Preview first few
        for t in synthetic.texts[:3]:
            print(f"\n  [{t.text_id}] Labels: {t.source_labels}")
            print(f"    Text: {t.text[:80]}...")

        # Step 4: Validate synthetic data
        print("\n" + "=" * 50)
        print("Step 4: Validation")
        print("=" * 50)
        validation = run_validation(
            synthetic_batch=synthetic,
            processor=processor,
            tasks=["alerts", "recommendations"],
            match_threshold=0.5,
            output_path="output/validation_batch_001.json",
        )

        print("\nValidation Summary:")
        print(f"  Total texts: {validation.total_texts}")
        print(f"  Valid: {validation.valid_count}")
        print(f"  Validation rate: {validation.validation_rate:.1%}")
        print(f"  Avg match ratio: {validation.avg_match_ratio:.1%}")

        # Show some examples
        valid_texts = get_valid_texts(validation)
        invalid_texts = get_invalid_texts(validation)

        if valid_texts:
            print("\nExample valid text:")
            t = valid_texts[0]
            print(f"  [{t.text_id}] {t.text[:80]}...")
            print(f"  Expected: {t.label_match.expected}")
            print(f"  Detected: {t.label_match.detected}")

        if invalid_texts:
            print("\nExample invalid text:")
            t = invalid_texts[0]
            print(f"  [{t.text_id}] {t.text[:80]}...")
            print(f"  Expected: {t.label_match.expected}")
            print(f"  Detected: {t.label_match.detected}")
            print(f"  Missed: {t.label_match.missed}")

        print("\n" + "=" * 50)
        print("Pipeline Complete!")
        print("=" * 50)
        print("\nOutputs saved to:")
        print("  - output/batch_001.json (classification)")
        print("  - output/excerpt_bank.json (excerpt bank)")
        print("  - output/synthetic_batch_001.json (synthetic data)")
        print("  - output/validation_batch_001.json (validation results)")
