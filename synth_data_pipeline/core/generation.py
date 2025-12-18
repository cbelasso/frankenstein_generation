"""Generic generation module."""

import json
from typing import Dict, List, Optional, Type

from pydantic import BaseModel

from .data_store import DataStore
from ..prompts import (
    format_frankenstein_prompt,
    format_style_transfer_prompt,
    format_non_alert_prompt,
)
from ..schemas import GeneratedComment, GeneratedNonAlertComment, StyleTransferOutput


class Generator:
    """Generic generator for synthetic data."""
    
    def __init__(
        self,
        processor,  # NewProcessor instance
        data_store: DataStore,
        guided_config: Optional[Dict] = None,
    ):
        self.processor = processor
        self.data_store = data_store
        self.guided_config = guided_config or {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "max_tokens": 1500,
        }
    
    def generate_from_labels(
        self,
        target_labels: List[str],
        n_samples: int = 1,
        n_candidates_per_label: int = 5,
        comment_type: str = "alert",
        batch_size: int = 10,
    ) -> List[GeneratedComment]:
        """Generate comments targeting specific labels."""
        candidate_excerpts = self.data_store.get_excerpts_by_labels(
            labels=target_labels,
            n_per_label=n_candidates_per_label,
            random_sample=True,
        )
        
        # Filter out empty labels
        candidate_excerpts = {k: v for k, v in candidate_excerpts.items() if v}
        
        missing_labels = [l for l in target_labels if l not in candidate_excerpts]
        if missing_labels:
            print(f"Warning: No excerpts found for labels: {missing_labels}")
        
        if not candidate_excerpts:
            print("Error: No candidate excerpts available for any target label")
            return []
        
        prompts = [
            format_frankenstein_prompt(
                target_labels=target_labels,
                candidate_excerpts=candidate_excerpts,
                comment_type=comment_type,
            )
            for _ in range(n_samples)
        ]
        
        responses = self.processor.process_with_schema(
            prompts=prompts,
            schema=GeneratedComment,
            batch_size=batch_size,
            guided_config=self.guided_config,
        )
        
        results = self.processor.parse_results_with_schema(
            schema=GeneratedComment,
            responses=responses,
            validate=True,
        )
        
        return [r for r in results if r is not None]
    
    def generate_batch_from_label_specs(
        self,
        label_specs: List[List[str]],
        n_candidates_per_label: int = 5,
        comment_type: str = "alert",
        batch_size: int = 25,
    ) -> List[GeneratedComment]:
        """Generate comments for multiple label specifications."""
        prompts = []
        
        for target_labels in label_specs:
            candidate_excerpts = self.data_store.get_excerpts_by_labels(
                labels=target_labels,
                n_per_label=n_candidates_per_label,
                random_sample=True,
            )
            
            candidate_excerpts = {k: v for k, v in candidate_excerpts.items() if v}
            
            if not candidate_excerpts:
                print(f"Warning: No excerpts for spec {target_labels}, skipping")
                continue
            
            prompts.append(
                format_frankenstein_prompt(
                    target_labels=target_labels,
                    candidate_excerpts=candidate_excerpts,
                    comment_type=comment_type,
                )
            )
        
        if not prompts:
            return []
        
        responses = self.processor.process_with_schema(
            prompts=prompts,
            schema=GeneratedComment,
            batch_size=batch_size,
            guided_config=self.guided_config,
        )
        
        results = self.processor.parse_results_with_schema(
            schema=GeneratedComment,
            responses=responses,
            validate=True,
        )
        
        return results
    
    def generate_non_alerts(
        self,
        target_classification: str,
        n_samples: int = 1,
        n_references: int = 5,
        batch_size: int = 10,
    ) -> List[GeneratedNonAlertComment]:
        """Generate non-alert comments for a specific classification."""
        reference_comments = self.data_store.get_comments_by_negative_class(
            classification=target_classification,
            n=n_references,
            random_sample=True,
        )
        
        if not reference_comments:
            print(f"Warning: No reference comments found for '{target_classification}'")
            return []
        
        prompts = [
            format_non_alert_prompt(
                target_classification=target_classification,
                reference_comments=reference_comments,
            )
            for _ in range(n_samples)
        ]
        
        responses = self.processor.process_with_schema(
            prompts=prompts,
            schema=GeneratedNonAlertComment,
            batch_size=batch_size,
            guided_config=self.guided_config,
        )
        
        results = self.processor.parse_results_with_schema(
            schema=GeneratedNonAlertComment,
            responses=responses,
            validate=True,
        )
        
        return [r for r in results if r is not None]
    
    def generate_batch_non_alerts(
        self,
        classification_specs: List[str],
        n_references: int = 5,
        batch_size: int = 25,
    ) -> List[GeneratedNonAlertComment]:
        """Generate non-alert comments for multiple classifications."""
        prompts = []
        
        for classification in classification_specs:
            reference_comments = self.data_store.get_comments_by_negative_class(
                classification=classification,
                n=n_references,
                random_sample=True,
            )
            
            if not reference_comments:
                print(f"Warning: No references for '{classification}', skipping")
                continue
            
            prompts.append(
                format_non_alert_prompt(
                    target_classification=classification,
                    reference_comments=reference_comments,
                )
            )
        
        if not prompts:
            return []
        
        responses = self.processor.process_with_schema(
            prompts=prompts,
            schema=GeneratedNonAlertComment,
            batch_size=batch_size,
            guided_config=self.guided_config,
        )
        
        results = self.processor.parse_results_with_schema(
            schema=GeneratedNonAlertComment,
            responses=responses,
            validate=True,
        )
        
        return results
    
    def style_transfer(
        self,
        input_comments: List[str],
        reference_comments: List[str],
        batch_size: int = 25,
    ) -> List[str]:
        """Apply style transfer to comments."""
        if len(input_comments) != len(reference_comments):
            raise ValueError("input_comments and reference_comments must have same length")
        
        prompts = [
            format_style_transfer_prompt(ref, inp)
            for inp, ref in zip(input_comments, reference_comments)
        ]
        
        responses = self.processor.process_with_schema(
            prompts=prompts,
            schema=StyleTransferOutput,
            batch_size=batch_size,
            guided_config={
                "temperature": 0.3,
                "max_tokens": 1000,
            },
        )
        
        results = self.processor.parse_results_with_schema(
            schema=StyleTransferOutput,
            responses=responses,
            validate=True,
        )
        
        return [r.rewritten_comment if r else "" for r in results]


class BalancedGenerator:
    """Generator for balanced/minority class augmentation."""
    
    def __init__(self, generator: Generator):
        self.generator = generator
    
    def generate_for_minority_labels(
        self,
        target_count: int,
        min_samples_threshold: int = 10,
        n_candidates_per_label: int = 5,
    ) -> Dict[str, List[GeneratedComment]]:
        """Generate samples for underrepresented labels."""
        label_counts = self.generator.data_store.get_label_counts()
        
        minority_labels = [
            label for label, count in label_counts.items()
            if count < min_samples_threshold
        ]
        
        results = {}
        for label in minority_labels:
            current_count = label_counts[label]
            n_to_generate = max(0, target_count - current_count)
            
            if n_to_generate > 0:
                generated = self.generator.generate_from_labels(
                    target_labels=[label],
                    n_samples=n_to_generate,
                    n_candidates_per_label=n_candidates_per_label,
                )
                results[label] = generated
        
        return results
    
    def generate_for_rare_combinations(
        self,
        label_combinations: List[List[str]],
        n_per_combination: int = 5,
        n_candidates_per_label: int = 5,
    ) -> Dict[str, List[GeneratedComment]]:
        """Generate samples for rare label combinations."""
        results = {}
        
        for combo in label_combinations:
            combo_key = "+".join(sorted(combo))
            generated = self.generator.generate_from_labels(
                target_labels=combo,
                n_samples=n_per_combination,
                n_candidates_per_label=n_candidates_per_label,
            )
            results[combo_key] = generated
        
        return results
    
    def generate_for_minority_negative_classes(
        self,
        target_count: int,
        min_samples_threshold: int = 10,
        n_references: int = 5,
    ) -> Dict[str, List[GeneratedNonAlertComment]]:
        """Generate samples for underrepresented negative classifications."""
        class_counts = self.generator.data_store.get_negative_class_counts()
        
        minority_classes = [
            cls for cls, count in class_counts.items()
            if count < min_samples_threshold
        ]
        
        results = {}
        for classification in minority_classes:
            current_count = class_counts[classification]
            n_to_generate = max(0, target_count - current_count)
            
            if n_to_generate > 0:
                generated = self.generator.generate_non_alerts(
                    target_classification=classification,
                    n_samples=n_to_generate,
                    n_references=n_references,
                )
                results[classification] = generated
        
        return results
