import random

import numpy as np

from .inter import Article, Model, Prediction


class RandomModel(Model):
    """A model which generates random predictions."""

    def predict(self, article: Article) -> Prediction:
        # Use softmax to turn random values into probabilities.
        label_probs = [random.random() for _ in Prediction.LABELS]
        label_probs = np.array(label_probs)
        label_probs = np.exp(label_probs) / sum(np.exp(label_probs))

        # Generate a random probability for each category separately.
        category_probs = [random.random() for _ in Prediction.CATEGORIES]

        return Prediction(
            label_probs=dict(zip(Prediction.LABELS, label_probs)),
            category_probs=dict(zip(Prediction.CATEGORIES, category_probs)),
        )
