from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Article:
    title: Optional[str]
    body: str

    @classmethod
    def empty_article(cls) -> Article:
        return Article(title="", body="")


@dataclass
class Prediction:
    LABELS = [
        "știre fabricată",
        "știre ficțională",
        "știre plauzibilă",
        "știre propagandistică",
        "știre reală",
        "știre satirică",
    ]

    CATEGORIES = [
        "sănătate",
        "politică",
    ]

    label_probs: dict[str, float]
    category_probs: dict[str, float]

    @classmethod
    def empty_prediction(cls) -> Prediction:
        return Prediction(
            label_probs={label: 0.0 for label in Prediction.LABELS},
            category_probs={cat: 0.0 for cat in Prediction.CATEGORIES},
        )


class Model(ABC):
    @abstractmethod
    def predict(self, article: Article) -> Prediction:
        raise NotImplementedError("model does not implement prediction")
