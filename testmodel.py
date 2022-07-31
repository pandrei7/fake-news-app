from datasets import DatasetDict
import matplotlib.pyplot as plt
from huggingface_hub import from_pretrained_keras
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from transformers import AutoTokenizer, DefaultDataCollator

CATS_TO_SEARCH = ["sănătate", "politică"]
CAT_OUT_NAMES = {
    "sănătate": "out_sanatate",
    "politică": "out_politica",
    "religie": "out_religie",
}

INPUT_COLS = ["input_ids", "token_type_ids", "attention_mask"]
OUTPUT_COLS = ["labels"] + [CAT_OUT_NAMES[cat] for cat in CATS_TO_SEARCH]
BATCH_SIZE = 8


tokenizer = AutoTokenizer.from_pretrained("readerbench/RoBERT-base")
model = from_pretrained_keras("pandrei7/fakenews-mtl")
print(model.summary())


def prep_func(batch):
    return tokenizer(
        batch["article"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


ds = DatasetDict.load_from_disk("../dataset/hf/ds")
prep_ds = ds.map(prep_func, batched=True)
tf_test = prep_ds["test"].to_tf_dataset(
    columns=INPUT_COLS,
    label_cols=OUTPUT_COLS,
    shuffle=False,
    collate_fn=DefaultDataCollator(),
    batch_size=BATCH_SIZE,
)

predictions = model.predict(tf_test)
label_predictions = predictions["labels"].argmax(axis=-1)
true_pred = prep_ds["test"]["label"]
report = classification_report(true_pred, label_predictions, output_dict=True)
print(classification_report(true_pred, label_predictions))

ConfusionMatrixDisplay.from_predictions(
    true_pred,
    label_predictions,
    display_labels=prep_ds["train"].features["label"].names,
)
plt.xticks(rotation=25)
plt.show()
