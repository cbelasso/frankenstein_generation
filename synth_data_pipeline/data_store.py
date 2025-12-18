from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import pandas as pd
from pydantic import BaseModel

from .schemas import AlertsOutput, AlertSpan


class DataStore:
    def __init__(self):
        self.comments_df: Optional[pd.DataFrame] = None
        self.excerpts_df: Optional[pd.DataFrame] = None
        self.label_index: Dict[str, List[int]] = defaultdict(list)
        self.non_alert_index: Dict[str, List[int]] = defaultdict(list)

    def load_from_extraction_results(
        self,
        original_comments: List[str],
        extraction_results: List[AlertsOutput],
    ) -> None:
        comments_data = []
        excerpts_data = []

        for idx, (comment, result) in enumerate(zip(original_comments, extraction_results)):
            if result is None:
                continue

            alert_types = [a.alert_type for a in result.alerts] if result.alerts else []

            comments_data.append(
                {
                    "comment_id": idx,
                    "original_comment": comment,
                    "has_alerts": result.has_alerts,
                    "alert_types": alert_types,
                    "non_alert_classification": result.non_alert_classification,
                    "non_alert_reasoning": result.non_alert_reasoning,
                }
            )

            if result.has_alerts:
                for alert in result.alerts:
                    self.label_index[alert.alert_type].append(len(excerpts_data))
                    excerpts_data.append(
                        {
                            "excerpt_id": len(excerpts_data),
                            "comment_id": idx,
                            "excerpt": alert.excerpt,
                            "reasoning": alert.reasoning,
                            "alert_type": alert.alert_type,
                            "severity": alert.severity,
                        }
                    )

            if result.non_alert_classification:
                self.non_alert_index[result.non_alert_classification].append(idx)

        self.comments_df = pd.DataFrame(comments_data)
        self.excerpts_df = pd.DataFrame(excerpts_data)

    def get_excerpts_by_label(
        self,
        label: str,
        n: Optional[int] = None,
        random_sample: bool = False,
    ) -> List[str]:
        if self.excerpts_df is None or label not in self.label_index:
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
        return {
            label: self.get_excerpts_by_label(label, n=n_per_label, random_sample=random_sample)
            for label in labels
        }

    def get_comments_by_non_alert_class(
        self,
        classification: str,
        n: Optional[int] = None,
        random_sample: bool = False,
    ) -> List[str]:
        if self.comments_df is None or classification not in self.non_alert_index:
            return []

        indices = self.non_alert_index[classification]
        subset = self.comments_df[self.comments_df["comment_id"].isin(indices)]

        if random_sample and n:
            subset = subset.sample(n=min(n, len(subset)))
        elif n:
            subset = subset.head(n)

        return subset["original_comment"].tolist()

    def get_available_labels(self) -> List[str]:
        return list(self.label_index.keys())

    def get_available_non_alert_classes(self) -> List[str]:
        return list(self.non_alert_index.keys())

    def get_label_counts(self) -> Dict[str, int]:
        return {label: len(indices) for label, indices in self.label_index.items()}

    def get_non_alert_counts(self) -> Dict[str, int]:
        return {cls: len(indices) for cls, indices in self.non_alert_index.items()}

    def get_cooccurrence_matrix(self) -> pd.DataFrame:
        if self.comments_df is None:
            return pd.DataFrame()

        labels = self.get_available_labels()
        matrix = pd.DataFrame(0, index=labels, columns=labels)

        for _, row in self.comments_df.iterrows():
            alert_types = row["alert_types"]
            for i, l1 in enumerate(alert_types):
                for l2 in alert_types[i:]:
                    matrix.loc[l1, l2] += 1
                    if l1 != l2:
                        matrix.loc[l2, l1] += 1

        return matrix

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.comments_df is not None:
            self.comments_df.to_csv(path / "comments.csv", index=False)
        if self.excerpts_df is not None:
            self.excerpts_df.to_csv(path / "excerpts.csv", index=False)

        with open(path / "label_index.json", "w") as f:
            json.dump(dict(self.label_index), f, indent=2)
        with open(path / "non_alert_index.json", "w") as f:
            json.dump(dict(self.non_alert_index), f, indent=2)

    def load(self, path: str) -> None:
        path = Path(path)

        if (path / "comments.csv").exists():
            self.comments_df = pd.read_csv(path / "comments.csv")
            # Convert string representation of list back to list
            if "alert_types" in self.comments_df.columns:
                import ast

                self.comments_df["alert_types"] = self.comments_df["alert_types"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
        if (path / "excerpts.csv").exists():
            self.excerpts_df = pd.read_csv(path / "excerpts.csv")

        if (path / "label_index.json").exists():
            with open(path / "label_index.json") as f:
                self.label_index = defaultdict(list, json.load(f))
        if (path / "non_alert_index.json").exists():
            with open(path / "non_alert_index.json") as f:
                self.non_alert_index = defaultdict(list, json.load(f))
