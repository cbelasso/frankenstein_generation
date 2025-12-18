"""Data store for extracted data."""

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel

from ..schemas.base import ExtractionOutput


class DataStore:
    """Generic data store for extraction results."""
    
    def __init__(self):
        self.comments_df: Optional[pd.DataFrame] = None
        self.excerpts_df: Optional[pd.DataFrame] = None
        self.label_index: Dict[str, List[int]] = defaultdict(list)
        self.negative_class_index: Dict[str, List[int]] = defaultdict(list)
    
    def load_from_extraction_results(
        self,
        original_texts: List[str],
        extraction_results: List[ExtractionOutput],
        text_column: str = "original_comment",
    ) -> None:
        """Load data from extraction results (schema-agnostic)."""
        comments_data = []
        excerpts_data = []
        
        for idx, (text, result) in enumerate(zip(original_texts, extraction_results)):
            if result is None:
                continue
            
            labels = list(result.get_labels())
            
            comments_data.append({
                "comment_id": idx,
                text_column: text,
                "has_positive_labels": result.has_positive_labels(),
                "labels": labels,
                "negative_classification": result.get_negative_classification(),
            })
            
            if result.has_positive_labels():
                for excerpt_data in result.get_excerpts():
                    label = excerpt_data.get("label")
                    self.label_index[label].append(len(excerpts_data))
                    excerpts_data.append({
                        "excerpt_id": len(excerpts_data),
                        "comment_id": idx,
                        **excerpt_data,
                    })
            
            neg_class = result.get_negative_classification()
            if neg_class:
                self.negative_class_index[neg_class].append(idx)
        
        self.comments_df = pd.DataFrame(comments_data)
        self.excerpts_df = pd.DataFrame(excerpts_data) if excerpts_data else pd.DataFrame()
    
    def get_excerpts_by_label(
        self,
        label: str,
        n: Optional[int] = None,
        random_sample: bool = False,
    ) -> List[str]:
        """Get excerpts for a specific label."""
        if self.excerpts_df is None or self.excerpts_df.empty or label not in self.label_index:
            return []
        
        indices = self.label_index[label]
        subset = self.excerpts_df.iloc[indices]
        
        if random_sample and n:
            subset = subset.sample(n=min(n, len(subset)))
        elif n:
            subset = subset.head(n)
        
        return subset["excerpt"].tolist()
    
    def get_excerpts_by_labels(
        self,
        labels: List[str],
        n_per_label: int = 5,
        random_sample: bool = True,
    ) -> Dict[str, List[str]]:
        """Get excerpts for multiple labels."""
        return {
            label: self.get_excerpts_by_label(label, n=n_per_label, random_sample=random_sample)
            for label in labels
        }
    
    def get_comments_by_negative_class(
        self,
        classification: str,
        n: Optional[int] = None,
        random_sample: bool = False,
        text_column: str = "original_comment",
    ) -> List[str]:
        """Get comments for a specific negative classification."""
        if self.comments_df is None or classification not in self.negative_class_index:
            return []
        
        indices = self.negative_class_index[classification]
        subset = self.comments_df[self.comments_df["comment_id"].isin(indices)]
        
        if random_sample and n:
            subset = subset.sample(n=min(n, len(subset)))
        elif n:
            subset = subset.head(n)
        
        return subset[text_column].tolist()
    
    def get_available_labels(self) -> List[str]:
        """Get all available labels."""
        return list(self.label_index.keys())
    
    def get_available_negative_classes(self) -> List[str]:
        """Get all available negative classifications."""
        return list(self.negative_class_index.keys())
    
    def get_label_counts(self) -> Dict[str, int]:
        """Get count of excerpts per label."""
        return {label: len(indices) for label, indices in self.label_index.items()}
    
    def get_negative_class_counts(self) -> Dict[str, int]:
        """Get count of comments per negative classification."""
        return {cls: len(indices) for cls, indices in self.negative_class_index.items()}
    
    def get_cooccurrence_matrix(self) -> pd.DataFrame:
        """Get label co-occurrence matrix."""
        if self.comments_df is None:
            return pd.DataFrame()
        
        labels = self.get_available_labels()
        matrix = pd.DataFrame(0, index=labels, columns=labels)
        
        for _, row in self.comments_df.iterrows():
            row_labels = row["labels"]
            if isinstance(row_labels, str):
                row_labels = ast.literal_eval(row_labels)
            for i, l1 in enumerate(row_labels):
                for l2 in row_labels[i:]:
                    matrix.loc[l1, l2] += 1
                    if l1 != l2:
                        matrix.loc[l2, l1] += 1
        
        return matrix
    
    def save(self, path: str) -> None:
        """Save data store to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.comments_df is not None:
            self.comments_df.to_csv(path / "comments.csv", index=False)
        if self.excerpts_df is not None and not self.excerpts_df.empty:
            self.excerpts_df.to_csv(path / "excerpts.csv", index=False)
        
        with open(path / "label_index.json", "w") as f:
            json.dump(dict(self.label_index), f, indent=2)
        with open(path / "negative_class_index.json", "w") as f:
            json.dump(dict(self.negative_class_index), f, indent=2)
    
    def load(self, path: str) -> None:
        """Load data store from directory."""
        path = Path(path)
        
        if (path / "comments.csv").exists():
            self.comments_df = pd.read_csv(path / "comments.csv")
            if "labels" in self.comments_df.columns:
                self.comments_df["labels"] = self.comments_df["labels"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
        
        if (path / "excerpts.csv").exists():
            self.excerpts_df = pd.read_csv(path / "excerpts.csv")
        
        if (path / "label_index.json").exists():
            with open(path / "label_index.json") as f:
                self.label_index = defaultdict(list, json.load(f))
        
        if (path / "negative_class_index.json").exists():
            with open(path / "negative_class_index.json") as f:
                self.negative_class_index = defaultdict(list, json.load(f))


# Backwards compatibility alias
def get_available_non_alert_classes(store: DataStore) -> List[str]:
    """Alias for backwards compatibility."""
    return store.get_available_negative_classes()


def get_comments_by_non_alert_class(
    store: DataStore,
    classification: str,
    n: Optional[int] = None,
    random_sample: bool = False,
) -> List[str]:
    """Alias for backwards compatibility."""
    return store.get_comments_by_negative_class(classification, n, random_sample)
