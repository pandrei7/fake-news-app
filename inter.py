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
    probs: dict[str, float]

    @classmethod
    def empty_prediction(cls) -> Prediction:
        return Prediction({})


class Model(ABC):
    @abstractmethod
    def predict(self, article: Article) -> Prediction:
        pass
