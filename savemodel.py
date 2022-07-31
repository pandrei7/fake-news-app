from huggingface_hub import push_to_hub_keras
from transformers import PretrainedConfig

from models import BertBasedModel

NAME = "fakenews-mtl"

model = BertBasedModel()
model = model.model
config = {"min_length": 512, "max_length": 512}
push_to_hub_keras(model, NAME, config=config)
