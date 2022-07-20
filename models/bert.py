from huggingface_hub import from_pretrained_keras
import numpy as np
import transformers
from transformers import AutoTokenizer

from .inter import Article, Model, Prediction


class BertBasedModel(Model):
    TOKENIZER_CHECKPOINT = "readerbench/RoBERT-base"
    MODEL_CHECKPOINT = "pandrei7/fakenews-mtl"

    MAX_LENGTH = 512

    LABEL_ORDER = sorted(Prediction.LABELS)

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BertBasedModel.TOKENIZER_CHECKPOINT
        )
        self.model = from_pretrained_keras(BertBasedModel.MODEL_CHECKPOINT)

    def predict(self, article: Article) -> Prediction:
        tokens = self.prepare_inputs(article)
        output = self.model(tokens.data)

        label_probs = np.array(output["labels"][0], dtype=float)
        label_probs = dict(zip(BertBasedModel.LABEL_ORDER, label_probs))

        category_probs = {
            "sănătate": float(output["out_sanatate"][0]),
            "politică": float(output["out_politica"][0]),
        }

        return Prediction(
            label_probs=label_probs,
            category_probs=category_probs,
        )

    def prepare_inputs(self, article: Article) -> transformers.BatchEncoding:
        content = article.title + "\n[SEP]\n" + article.body

        return self.tokenizer(
            content,
            padding="max_length",
            truncation=True,
            max_length=BertBasedModel.MAX_LENGTH,
            return_tensors="tf",
        )
