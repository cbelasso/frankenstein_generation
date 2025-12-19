"""
Processor protocol definition.

Defines the interface that any processor must implement to work with pipeline capabilities.
This allows swapping between different backends (vLLM, API-based, etc.) without changing
the extraction logic.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Type, Union

from pydantic import BaseModel


class ProcessorProtocol(Protocol):
    """
    Protocol defining the interface for LLM processors.

    Any processor that implements these methods can be used with the pipeline.
    Your NewProcessor already conforms to this interface.
    """

    def process_with_schema(
        self,
        prompts: Union[str, List[str]],
        schema: Optional[Type[BaseModel]] = None,
        batch_size: int = 25,
        formatted: bool = False,
        guided_config: Optional[Dict] = None,
        on_batch_end: Optional[Callable] = None,
        timeout: int = 10,
    ) -> List[Any]:
        """
        Process prompts with optional JSON schema enforcement.

        Args:
            prompts: Single prompt or list of prompts to process
            schema: Pydantic model to use for structured output
            batch_size: Number of prompts per batch
            formatted: Whether prompts are already formatted with chat template
            guided_config: Optional config for guided decoding parameters
            on_batch_end: Optional callback after each batch completes
            timeout: Timeout for queue operations

        Returns:
            List of response objects (RequestOutput from vLLM or similar)
        """
        ...

    def parse_results_with_schema(
        self,
        schema: Type[BaseModel],
        responses: Optional[List[Any]] = None,
        validate: bool = True,
    ) -> List[Union[BaseModel, Dict, str, None]]:
        """
        Parse response outputs into Pydantic models.

        Args:
            schema: Pydantic model class to parse into
            responses: List of response objects (uses self.responses if None)
            validate: Whether to validate against the schema

        Returns:
            List of parsed objects (Pydantic models if validate=True, else dicts)
        """
        ...

    def format_prompt(self, prompt: str) -> str:
        """
        Format a prompt using the model's chat template.

        Args:
            prompt: Raw prompt text

        Returns:
            Formatted prompt string
        """
        ...
