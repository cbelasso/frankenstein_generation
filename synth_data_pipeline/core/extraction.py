"""Generic extraction module."""

from typing import Dict, List, Optional, Type

from pydantic import BaseModel

from ..prompts.base import PromptTemplate


class Extractor:
    """Generic extractor that works with any schema and prompt."""
    
    def __init__(
        self,
        processor,  # NewProcessor instance
        schema: Type[BaseModel],
        prompt_template: PromptTemplate,
        guided_config: Optional[Dict] = None,
    ):
        self.processor = processor
        self.schema = schema
        self.prompt_template = prompt_template
        self.guided_config = guided_config or {
            "temperature": 0.1,
            "top_k": 50,
            "top_p": 0.95,
            "max_tokens": 2000,
        }
    
    def extract(
        self,
        texts: List[str],
        batch_size: int = 25,
        validate: bool = True,
    ) -> List[BaseModel]:
        """Extract from a list of texts."""
        prompts = [self.prompt_template.format(text=text) for text in texts]
        
        responses = self.processor.process_with_schema(
            prompts=prompts,
            schema=self.schema,
            batch_size=batch_size,
            guided_config=self.guided_config,
        )
        
        results = self.processor.parse_results_with_schema(
            schema=self.schema,
            responses=responses,
            validate=validate,
        )
        
        return results
    
    def extract_single(self, text: str, validate: bool = True) -> BaseModel:
        """Extract from a single text."""
        results = self.extract([text], batch_size=1, validate=validate)
        return results[0] if results else None


def create_extractor(
    processor,
    schema: Type[BaseModel],
    prompt_template: PromptTemplate,
    guided_config: Optional[Dict] = None,
) -> Extractor:
    """Factory function to create an extractor."""
    return Extractor(
        processor=processor,
        schema=schema,
        prompt_template=prompt_template,
        guided_config=guided_config,
    )


# Convenience class for alerts (backwards compatibility)
class AlertsExtractor(Extractor):
    """Pre-configured extractor for alerts."""
    
    def __init__(self, processor, guided_config: Optional[Dict] = None):
        from ..schemas import AlertsOutput
        from ..prompts import ALERTS_EXTRACTION_PROMPT
        
        super().__init__(
            processor=processor,
            schema=AlertsOutput,
            prompt_template=ALERTS_EXTRACTION_PROMPT,
            guided_config=guided_config,
        )
