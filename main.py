#!/usr/bin/env python3
"""
Config-driven synthetic data generation pipeline.

Usage:
    # With config file
    python main.py --config config/alerts.yaml
    
    # With config + CLI overrides
    python main.py --config config/alerts.yaml --input data.csv --column text --gpus 0 1 2 3
    
    # Quick run without config
    python main.py --input data.csv --column comment_text --gpus 0
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from llm_parallelization.new_processor import NewProcessor
from synth_data_pipeline import (
    PipelineConfig,
    DataStore,
    Extractor,
    Generator,
    Validator,
    print_validation_result,
    get_schema,
    get_prompt,
)


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
    config: PipelineConfig,
    texts: List[str],
) -> tuple[DataStore, list]:
    """Run extraction on texts."""
    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTION")
    print("=" * 60)
    
    schema = config.get_extraction_schema()
    prompt = config.get_extraction_prompt()
    
    print(f"Schema: {schema.__name__}")
    print(f"Prompt: {prompt.name}")
    
    extractor = Extractor(
        processor=processor,
        schema=schema,
        prompt_template=prompt,
        guided_config=config.extraction_guided_config,
    )
    
    extractions = extractor.extract(texts, batch_size=config.batch_size)
    
    store = DataStore()
    store.load_from_extraction_results(texts, extractions)
    
    print(f"\n✓ Extracted {len(store.excerpts_df) if store.excerpts_df is not None else 0} excerpts from {len(texts)} comments")
    print(f"✓ Labels found: {store.get_available_labels()}")
    print(f"✓ Label counts: {store.get_label_counts()}")
    
    return store, extractions


def run_generation(
    processor: NewProcessor,
    config: PipelineConfig,
    store: DataStore,
) -> tuple[list, list]:
    """Run generation based on config."""
    all_alerts = []
    all_non_alerts = []
    
    generator = Generator(
        processor=processor,
        data_store=store,
        guided_config=config.generation_guided_config,
    )
    
    gen_type = config.generation_type
    
    # Generate alerts
    if gen_type in ["alerts", "both"]:
        print("\n" + "=" * 60)
        print("STEP 2a: ALERT GENERATION")
        print("=" * 60)
        
        target_labels = config.target_labels
        if not target_labels:
            available = store.get_available_labels()
            target_labels = available[:2] if len(available) >= 2 else available
        
        if target_labels:
            # Verify labels exist
            available = set(store.get_available_labels())
            valid_labels = [l for l in target_labels if l in available]
            missing = [l for l in target_labels if l not in available]
            
            if missing:
                print(f"⚠ Warning: Labels not in data: {missing}")
            
            if valid_labels:
                print(f"\nTarget labels: {valid_labels}")
                print(f"Generating {config.n_samples} samples...")
                
                generated = generator.generate_from_labels(
                    target_labels=valid_labels,
                    n_samples=config.n_samples,
                    n_candidates_per_label=config.n_candidates,
                )
                
                print(f"\n✓ Generated {len(generated)} alert comments")
                for i, g in enumerate(generated, 1):
                    print(f"\n[{i}] {g.comment[:100]}...")
                
                all_alerts.extend(generated)
        else:
            print("⚠ No alert labels found in data")
    
    # Generate non-alerts
    if gen_type in ["non-alerts", "both"]:
        print("\n" + "=" * 60)
        print("STEP 2b: NON-ALERT GENERATION")
        print("=" * 60)
        
        target_classes = config.target_classifications
        if not target_classes:
            available = store.get_available_negative_classes()
            target_classes = available[:2] if len(available) >= 2 else available
        
        for classification in target_classes:
            available = set(store.get_available_negative_classes())
            if classification not in available:
                print(f"⚠ Warning: Classification '{classification}' not in data")
                continue
            
            print(f"\nTarget classification: {classification}")
            print(f"Generating {config.n_samples} samples...")
            
            generated = generator.generate_non_alerts(
                target_classification=classification,
                n_samples=config.n_samples,
                n_references=config.n_candidates,
            )
            
            print(f"\n✓ Generated {len(generated)} non-alert comments")
            for i, g in enumerate(generated, 1):
                print(f"\n[{i}] {g.comment[:100]}...")
            
            all_non_alerts.extend(generated)
    
    return all_alerts, all_non_alerts


def run_validation(
    processor: NewProcessor,
    config: PipelineConfig,
    generated: list,
) -> None:
    """Validate generated comments."""
    print("\n" + "=" * 60)
    print("STEP 3: VALIDATION")
    print("=" * 60)
    
    schema = config.get_extraction_schema()
    prompt = config.get_extraction_prompt()
    
    extractor = Extractor(
        processor=processor,
        schema=schema,
        prompt_template=prompt,
        guided_config=config.extraction_guided_config,
    )
    
    validator = Validator(extractor, match_threshold=config.match_threshold)
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


def build_config_overrides(args) -> Dict:
    """Build config overrides from CLI args."""
    overrides = {}
    
    if args.input:
        overrides.setdefault("input", {})["file"] = args.input
    if args.column:
        overrides.setdefault("input", {})["column"] = args.column
    if args.max_samples:
        overrides.setdefault("input", {})["max_samples"] = args.max_samples
    
    if args.gpus:
        overrides.setdefault("processor", {})["gpus"] = args.gpus
    if args.model:
        overrides.setdefault("processor", {})["model"] = args.model
    if args.batch_size:
        overrides.setdefault("processor", {})["batch_size"] = args.batch_size
    
    if args.generate:
        overrides.setdefault("generation", {})["type"] = args.generate
    if args.n_samples:
        overrides.setdefault("generation", {})["n_samples"] = args.n_samples
    if args.n_candidates:
        overrides.setdefault("generation", {})["n_candidates"] = args.n_candidates
    if args.target_labels:
        overrides.setdefault("generation", {}).setdefault("alert", {})["target_labels"] = args.target_labels
    if args.target_classification:
        overrides.setdefault("generation", {}).setdefault("non_alert", {})["target_classifications"] = [args.target_classification]
    
    if args.skip_generation:
        overrides.setdefault("generation", {})["enabled"] = False
    if args.skip_validation:
        overrides.setdefault("validation", {})["enabled"] = False
    
    if args.output_dir:
        overrides.setdefault("output", {})["dir"] = args.output_dir
    
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Config-driven synthetic data pipeline")
    
    # Config file
    parser.add_argument("--config", "-cfg", default=None, help="Path to config YAML file")
    
    # Input overrides
    parser.add_argument("--input", "-i", default=None, help="Input file (CSV, Excel, or JSON)")
    parser.add_argument("--column", "-c", default=None, help="Column name containing text data")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    
    # Processor overrides
    parser.add_argument("--gpus", "-g", nargs="+", type=int, default=None, help="GPU IDs to use")
    parser.add_argument("--model", "-m", default=None, help="Model path")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    
    # Generation overrides
    parser.add_argument("--generate", choices=["alerts", "non-alerts", "both"], default=None)
    parser.add_argument("--target-labels", "-t", nargs="+", default=None)
    parser.add_argument("--target-classification", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--n-candidates", type=int, default=None)
    
    # Skip flags
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    
    # Output
    parser.add_argument("--output-dir", "-o", default=None)
    
    args = parser.parse_args()
    
    # Build config
    overrides = build_config_overrides(args)
    
    if args.config:
        config = PipelineConfig.from_file(args.config, overrides)
    else:
        # Default config with overrides
        default_config = {
            "task": {"name": "default", "description": "Default pipeline run"},
            "processor": {
                "model": "casperhansen/mistral-nemo-instruct-2407-awq",
                "gpus": [0],
                "batch_size": 25,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 4096,
            },
            "extraction": {
                "schema": "alerts.AlertsOutput",
                "prompt": "alerts.EXTRACTION_PROMPT",
            },
            "generation": {
                "enabled": True,
                "type": "alerts",
                "n_samples": 5,
                "n_candidates": 5,
            },
            "validation": {"enabled": True, "match_threshold": 0.5},
            "output": {"dir": "./output"},
        }
        from synth_data_pipeline.config import deep_merge
        config = PipelineConfig.from_dict(deep_merge(default_config, overrides))
    
    # Validate required inputs
    if not config.input_file:
        parser.error("--input is required (via config or CLI)")
    if not config.input_column:
        parser.error("--column is required (via config or CLI)")
    
    print(f"✓ Config: {config}")
    
    # Load data
    texts = load_data(config.input_file, config.input_column)
    
    if config.max_samples:
        texts = texts[:config.max_samples]
        print(f"✓ Limited to {len(texts)} samples")
    
    # Initialize processor
    print(f"\n✓ Initializing processor on GPUs: {config.gpus}")
    print(f"✓ Model: {config.model}")
    
    with NewProcessor(
        gpu_list=config.gpus,
        llm=config.model,
        multiplicity=config.multiplicity,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
    ) as processor:
        
        # Step 1: Extraction
        store, extractions = run_extraction(processor, config, texts)
        
        # Save extraction results
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        store.save(output_dir / "data_store")
        print(f"\n✓ Saved data store to {output_dir / 'data_store'}")
        
        if not config.generation_enabled:
            print("\n✓ Skipping generation (disabled in config)")
            return
        
        # Step 2: Generation
        all_alerts, all_non_alerts = run_generation(processor, config, store)
        
        # Save generated data
        if all_alerts:
            alerts_data = [g.model_dump() for g in all_alerts if g]
            with open(output_dir / "generated_alerts.json", "w") as f:
                json.dump(alerts_data, f, indent=2)
            print(f"\n✓ Saved {len(alerts_data)} alerts to {output_dir / 'generated_alerts.json'}")
        
        if all_non_alerts:
            non_alerts_data = [g.model_dump() for g in all_non_alerts if g]
            with open(output_dir / "generated_non_alerts.json", "w") as f:
                json.dump(non_alerts_data, f, indent=2)
            print(f"\n✓ Saved {len(non_alerts_data)} non-alerts to {output_dir / 'generated_non_alerts.json'}")
        
        # Step 3: Validation
        if config.validation_enabled and all_alerts:
            run_validation(processor, config, all_alerts)
        elif not config.validation_enabled:
            print("\n✓ Skipping validation (disabled in config)")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
