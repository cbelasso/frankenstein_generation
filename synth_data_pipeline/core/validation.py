"""Validation module for generated data."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

from pydantic import BaseModel

from .extraction import Extractor
from ..schemas.base import ExtractionOutput, GeneratedOutput


@dataclass
class ValidationResult:
    """Result of validating a generated comment."""
    
    generated_comment: str
    target_labels: Set[str]
    predicted_labels: Set[str]
    matched_labels: Set[str]
    missed_labels: Set[str]
    extra_labels: Set[str]
    precision: float
    recall: float
    f1: float
    passed: bool
    extraction_result: Optional[BaseModel] = None


class Validator:
    """Validates generated comments by re-extracting and comparing."""
    
    def __init__(
        self,
        extractor: Extractor,
        match_threshold: float = 0.5,
    ):
        self.extractor = extractor
        self.match_threshold = match_threshold
    
    def validate_single(
        self,
        generated: GeneratedOutput,
        label_extractor: Optional[Callable] = None,
    ) -> ValidationResult:
        """Validate a single generated comment."""
        extraction_result = self.extractor.extract_single(
            generated.comment,
            validate=True,
        )
        
        target_labels = set(generated.get_target_labels())
        
        if label_extractor:
            predicted_labels = set(label_extractor(extraction_result))
        else:
            predicted_labels = self._default_label_extractor(extraction_result)
        
        matched = target_labels & predicted_labels
        missed = target_labels - predicted_labels
        extra = predicted_labels - target_labels
        
        precision = len(matched) / len(predicted_labels) if predicted_labels else 0.0
        recall = len(matched) / len(target_labels) if target_labels else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        passed = recall >= self.match_threshold
        
        return ValidationResult(
            generated_comment=generated.comment,
            target_labels=target_labels,
            predicted_labels=predicted_labels,
            matched_labels=matched,
            missed_labels=missed,
            extra_labels=extra,
            precision=precision,
            recall=recall,
            f1=f1,
            passed=passed,
            extraction_result=extraction_result,
        )
    
    def validate_batch(
        self,
        generated_comments: List[GeneratedOutput],
        label_extractor: Optional[Callable] = None,
    ) -> List[ValidationResult]:
        """Validate a batch of generated comments."""
        comments = [g.comment for g in generated_comments]
        
        extraction_results = self.extractor.extract(comments, validate=True)
        
        results = []
        for generated, extraction_result in zip(generated_comments, extraction_results):
            target_labels = set(generated.get_target_labels())
            
            if label_extractor:
                predicted_labels = set(label_extractor(extraction_result))
            else:
                predicted_labels = self._default_label_extractor(extraction_result)
            
            matched = target_labels & predicted_labels
            missed = target_labels - predicted_labels
            extra = predicted_labels - target_labels
            
            precision = len(matched) / len(predicted_labels) if predicted_labels else 0.0
            recall = len(matched) / len(target_labels) if target_labels else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            passed = recall >= self.match_threshold
            
            results.append(ValidationResult(
                generated_comment=generated.comment,
                target_labels=target_labels,
                predicted_labels=predicted_labels,
                matched_labels=matched,
                missed_labels=missed,
                extra_labels=extra,
                precision=precision,
                recall=recall,
                f1=f1,
                passed=passed,
                extraction_result=extraction_result,
            ))
        
        return results
    
    def _default_label_extractor(self, result: BaseModel) -> Set[str]:
        """Extract labels from result using schema interface."""
        if result is None:
            return set()
        if hasattr(result, "get_labels"):
            return result.get_labels()
        return set()
    
    def summary(self, results: List[ValidationResult]) -> Dict:
        """Generate summary statistics for validation results."""
        if not results:
            return {}
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        
        avg_precision = sum(r.precision for r in results) / total
        avg_recall = sum(r.recall for r in results) / total
        avg_f1 = sum(r.f1 for r in results) / total
        
        all_missed = {}
        all_extra = {}
        
        for r in results:
            for label in r.missed_labels:
                all_missed[label] = all_missed.get(label, 0) + 1
            for label in r.extra_labels:
                all_extra[label] = all_extra.get(label, 0) + 1
        
        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "commonly_missed_labels": dict(sorted(all_missed.items(), key=lambda x: -x[1])[:10]),
            "commonly_extra_labels": dict(sorted(all_extra.items(), key=lambda x: -x[1])[:10]),
        }


def print_validation_result(result: ValidationResult) -> None:
    """Pretty-print a validation result."""
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"\n{status}")
    print(f"Comment: {result.generated_comment[:100]}...")
    print(f"Target:    {sorted(result.target_labels)}")
    print(f"Predicted: {sorted(result.predicted_labels)}")
    print(f"Matched:   {sorted(result.matched_labels)}")
    if result.missed_labels:
        print(f"Missed:    {sorted(result.missed_labels)}")
    if result.extra_labels:
        print(f"Extra:     {sorted(result.extra_labels)}")
    print(f"Precision: {result.precision:.2f} | Recall: {result.recall:.2f} | F1: {result.f1:.2f}")
