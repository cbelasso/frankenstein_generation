"""
Task runner.

Executes tasks using a processor, handling input validation and output parsing.
"""

from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from .base import BaseTask, get_task
from .models import TextInput


def run_task(
    task: Union[str, Type[BaseTask]],
    inputs: List[Union[Dict, BaseModel, str]],
    processor: Any,
    config_overrides: Optional[Dict[str, Any]] = None,
    batch_size: int = 25,
) -> List[BaseModel]:
    """
    Run a task on a list of inputs using the provided processor.

    Args:
        task: Task name (str) or task class
        inputs: List of inputs. Can be:
            - Dicts matching the task's input_model
            - Pydantic models matching the task's input_model
            - Strings (auto-wrapped in TextInput if task expects TextInput)
        processor: Processor instance (must implement ProcessorProtocol)
        config_overrides: Optional overrides for sampling config
        batch_size: Batch size for processing

    Returns:
        List of parsed output models
    """
    # Resolve task if string
    task_cls = get_task(task) if isinstance(task, str) else task

    # Validate and normalize inputs
    validated_inputs = _validate_inputs(task_cls, inputs)

    # Build prompts
    prompts = _build_prompts(task_cls, validated_inputs)

    # Merge config
    config = {**task_cls.get_config(), **(config_overrides or {})}

    # Run through processor
    responses = processor.process_with_schema(
        prompts=prompts,
        schema=task_cls.output_model,
        batch_size=batch_size,
        guided_config=config,
    )

    # Parse results
    results = processor.parse_results_with_schema(
        schema=task_cls.output_model,
        responses=responses,
        validate=True,
    )

    return results


def _validate_inputs(
    task_cls: Type[BaseTask], inputs: List[Union[Dict, BaseModel, str]]
) -> List[BaseModel]:
    """Validate and normalize inputs to the task's input_model."""
    validated = []

    for inp in inputs:
        if isinstance(inp, str):
            # Auto-wrap strings in TextInput if that's what the task expects
            if task_cls.input_model == TextInput:
                validated.append(TextInput(text=inp))
            else:
                raise TypeError(
                    f"Task '{task_cls.name}' expects {task_cls.input_model.__name__}, "
                    f"not a raw string. Wrap your input appropriately."
                )
        elif isinstance(inp, dict):
            validated.append(task_cls.input_model(**inp))
        elif isinstance(inp, task_cls.input_model):
            validated.append(inp)
        elif isinstance(inp, BaseModel):
            # Try to convert from another model
            validated.append(task_cls.input_model(**inp.model_dump()))
        else:
            raise TypeError(
                f"Invalid input type: {type(inp)}. "
                f"Expected {task_cls.input_model.__name__}, dict, or str."
            )

    return validated


def _build_prompts(task_cls: Type[BaseTask], inputs: List[BaseModel]) -> List[str]:
    """Build prompts from validated inputs."""
    prompts = []

    for inp in inputs:
        # Check if prompt_fn expects the model or just the text
        if task_cls.input_model == TextInput:
            # For TextInput, pass just the text string for backwards compatibility
            prompts.append(task_cls.prompt_fn(inp.text))
        else:
            # For other input models, pass the full model
            prompts.append(task_cls.prompt_fn(inp))

    return prompts
