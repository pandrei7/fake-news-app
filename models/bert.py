from pathlib import Path

import numpy as np
import tensorflow as tf
import transformers
from transformers import AutoTokenizer, TFAutoModel

from .inter import Article, Model, Prediction


class BertBasedModel(Model):
    TOKENIZER_CHECKPOINT = "readerbench/RoBERT-base"
    BERT_CHECKPOINT = Path(__file__).parent.parent / "bert" / "hf"
    TF_MODEL_PATH = Path(__file__).parent.parent / "bert" / "model.h5"

    MAX_LENGTH = 512

    LABEL_ORDER = sorted(Prediction.LABELS)

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            BertBasedModel.TOKENIZER_CHECKPOINT
        )
        self.bert = TFAutoModel.from_pretrained(BertBasedModel.BERT_CHECKPOINT)

        self.model = tf.keras.models.load_model(
            BertBasedModel.TF_MODEL_PATH,
            custom_objects={
                "TFBertModel": self.bert,
            },
            # Only load the model for prediction. We cannot load the model
            # without this option, because we miss a custom loss function.
            compile=False,
        )

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
