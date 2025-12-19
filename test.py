"""
Test script to verify pipeline imports and basic functionality.

Run with: python test_pipeline.py
"""


def test_imports():
    """Test all major imports work."""
    print("Testing imports...")

    # Core task system
    from pipeline.tasks import (
        BaseTask,
        TextInput,
        get_all_tasks,
        get_task,
        list_tasks,
        run_task,
    )

    print("  ✓ pipeline.tasks")

    # Task input models
    from pipeline.tasks.models import (
        ExcerptSetInput,
        TextInput,
        TextPairInput,
    )

    print("  ✓ pipeline.tasks.models")

    # Classification tasks
    from pipeline.tasks.classification import (
        ExcerptBank,
        ExcerptReference,
        ExtractedText,
        ExtractionBatch,
        ExtractionMetadata,
        alerts,
        recommendations,
    )

    print("  ✓ pipeline.tasks.classification")

    # Individual task modules
    from pipeline.tasks.classification.alerts import (
        AlertsOutput,
        AlertSpan,
        AlertsTask,
        alert_detection_prompt,
    )

    print("  ✓ pipeline.tasks.classification.alerts")

    from pipeline.tasks.classification.recommendations import (
        RecommendationsOutput,
        RecommendationSpan,
        RecommendationsTask,
        recommendations_detection_prompt,
    )

    print("  ✓ pipeline.tasks.classification.recommendations")

    # IO
    from pipeline.io import TextRecord, load_texts_from_csv

    print("  ✓ pipeline.io")

    # Processors
    from pipeline.processors import ProcessorProtocol

    print("  ✓ pipeline.processors")

    print()
    return True


def test_task_registry():
    """Test task registration and retrieval."""
    print("Testing task registry...")

    from pipeline.tasks import (
        classification,  # trigger registration
        get_all_tasks,
        get_task,
        list_tasks,
    )

    # List tasks
    all_tasks = list_tasks()
    assert "alerts" in all_tasks, "alerts task not registered"
    assert "recommendations" in all_tasks, "recommendations task not registered"
    print(f"  ✓ Registered tasks: {all_tasks}")

    # Filter by category
    classification_tasks = list_tasks(category="classification")
    assert classification_tasks == all_tasks, "category filter failed"
    print(f"  ✓ Classification tasks: {classification_tasks}")

    # Get task
    alerts_task = get_task("alerts")
    assert alerts_task.name == "alerts"
    assert alerts_task.category == "classification"
    print(f"  ✓ get_task('alerts'): {alerts_task.name}")

    # Task attributes
    from pipeline.tasks.classification.alerts import AlertsOutput
    from pipeline.tasks.models import TextInput

    assert alerts_task.input_model == TextInput
    assert alerts_task.output_model == AlertsOutput
    assert alerts_task.prompt_fn is not None
    assert alerts_task.default_config == {"temperature": 0.1, "max_tokens": 1000}
    print("  ✓ Task attributes validated")

    print()
    return True


def test_prompt_generation():
    """Test prompt generation for tasks."""
    print("Testing prompt generation...")

    from pipeline.tasks import classification, get_task

    # Alerts prompt
    alerts_task = get_task("alerts")
    prompt = alerts_task.prompt_fn("My manager yelled at me in front of everyone")
    assert len(prompt) > 1000, "prompt too short"
    assert "My manager yelled at me" in prompt, "input text not in prompt"
    print(f"  ✓ Alerts prompt: {len(prompt)} chars")

    # Recommendations prompt
    recs_task = get_task("recommendations")
    prompt = recs_task.prompt_fn("We need more hands-on exercises in the training")
    assert len(prompt) > 1000, "prompt too short"
    assert "hands-on exercises" in prompt, "input text not in prompt"
    print(f"  ✓ Recommendations prompt: {len(prompt)} chars")

    print()
    return True


def test_models():
    """Test Pydantic models."""
    print("Testing models...")

    from pipeline.tasks.classification import (
        ExcerptBank,
        ExcerptReference,
        ExtractedText,
        ExtractionBatch,
        ExtractionMetadata,
    )
    from pipeline.tasks.classification.alerts import AlertsOutput, AlertSpan
    from pipeline.tasks.models import ExcerptSetInput, TextInput, TextPairInput

    # TextInput
    inp = TextInput(text="Hello world")
    assert inp.text == "Hello world"
    assert inp.text_id is None
    print("  ✓ TextInput")

    # TextInput with optional fields
    inp = TextInput(text="Hello", text_id="t1", metadata={"source": "test"})
    assert inp.text_id == "t1"
    assert inp.metadata["source"] == "test"
    print("  ✓ TextInput with metadata")

    # AlertSpan
    span = AlertSpan(
        excerpt="called me a slur",
        reasoning="Discrimination",
        alert_type="discrimination",
        severity="high",
    )
    assert span.alert_type == "discrimination"
    print("  ✓ AlertSpan")

    # AlertsOutput
    output = AlertsOutput(has_alerts=True, alerts=[span])
    assert output.has_alerts is True
    assert len(output.alerts) == 1
    print("  ✓ AlertsOutput")

    # ExtractionBatch
    batch = ExtractionBatch(
        batch_id="test_batch",
        texts=[
            ExtractedText(
                text_id="t1",
                original_text="Test text",
                results={"alerts": output.model_dump()},
            )
        ],
    )
    assert batch.batch_id == "test_batch"
    assert len(batch.texts) == 1
    print("  ✓ ExtractionBatch")

    # ExcerptBank
    bank = ExcerptBank()
    bank.add_excerpt(
        "discrimination",
        ExcerptReference(
            excerpt="called me a slur",
            source_text_id="t1",
            task="alerts",
            label="discrimination",
        ),
    )
    assert "discrimination" in bank.list_labels()
    assert bank.count_by_label()["discrimination"] == 1
    print("  ✓ ExcerptBank")

    print()
    return True


def test_csv_reader():
    """Test CSV reader functionality."""
    print("Testing CSV reader...")

    import csv
    from pathlib import Path
    import tempfile

    from pipeline.io import TextRecord, load_texts_from_csv

    # Create temp CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(["id", "comment", "category"])
        writer.writerow(["1", "Great training!", "positive"])
        writer.writerow(["2", "Need more examples", "recommendation"])
        writer.writerow(["3", "", "empty"])  # Should be skipped
        writer.writerow(["4", "   ", "whitespace"])  # Should be skipped
        temp_path = f.name

    try:
        # Load with auto IDs
        records = load_texts_from_csv(temp_path, text_column="comment")
        assert len(records) == 2, f"Expected 2 records, got {len(records)}"
        assert records[0].text_id == "text_0001"
        assert records[0].text == "Great training!"
        print(f"  ✓ Auto-generated IDs: {len(records)} records")

        # Load with ID column
        records = load_texts_from_csv(temp_path, text_column="comment", id_column="id")
        assert len(records) == 2
        assert records[0].text_id == "1"
        print(f"  ✓ With ID column: {len(records)} records")

        # Custom prefix
        records = load_texts_from_csv(temp_path, text_column="comment", id_prefix="comment")
        assert records[0].text_id == "comment_0001"
        print(f"  ✓ Custom prefix: {records[0].text_id}")

    finally:
        Path(temp_path).unlink()

    print()
    return True


def test_json_serialization():
    """Test JSON save/load for models."""
    print("Testing JSON serialization...")

    from pathlib import Path
    import tempfile

    from pipeline.tasks.classification import (
        ExcerptBank,
        ExcerptReference,
        ExtractedText,
        ExtractionBatch,
    )

    # ExtractionBatch save/load
    batch = ExtractionBatch(
        batch_id="test",
        texts=[ExtractedText(text_id="t1", original_text="Hello", results={"test": "data"})],
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        batch.save(temp_path)
        loaded = ExtractionBatch.load(temp_path)
        assert loaded.batch_id == "test"
        assert len(loaded.texts) == 1
        assert loaded.texts[0].results["test"] == "data"
        print("  ✓ ExtractionBatch save/load")
    finally:
        Path(temp_path).unlink()

    # ExcerptBank save/load
    bank = ExcerptBank()
    bank.add_excerpt(
        "test_label",
        ExcerptReference(
            excerpt="test excerpt",
            source_text_id="t1",
            task="test",
            label="test_label",
        ),
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        bank.save(temp_path)
        loaded = ExcerptBank.load(temp_path)
        assert "test_label" in loaded.list_labels()
        assert loaded.count_by_label()["test_label"] == 1
        print("  ✓ ExcerptBank save/load")
    finally:
        Path(temp_path).unlink()

    print()
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Pipeline Test Suite")
    print("=" * 50)
    print()

    tests = [
        test_imports,
        test_task_registry,
        test_prompt_generation,
        test_models,
        test_csv_reader,
        test_json_serialization,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            print()

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
