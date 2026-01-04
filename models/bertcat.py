from typing import Dict, Union

import torch
from torch import nn as nn

from transformers import AutoModel
from transformers import PreTrainedModel, PretrainedConfig


class BERT_Cat_Config(PretrainedConfig):
    model_type = "BERT_Cat"
    bert_model: str
    trainable: bool = False


class BERTCat(PreTrainedModel):
    """
    Huggingface LM (bert,distillbert,roberta,albert) model for concated q-d cls scoring
    """

    config_class = BERT_Cat_Config
    base_model_prefix = "bert_model"

    @staticmethod
    def from_config(config):
        cfg = BERT_Cat_Config()
        cfg.bert_model = config["bert_pretrained_model"]
        cfg.trainable = config["bert_trainable"]
        return BERTCat(cfg)

    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)
        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)
        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self._classification_layer = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, _, tokens: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        pooled = self.forward_representation(tokens)
        score = self._classification_layer(pooled).squeeze(-1)
        return score

    def forward_representation(self,  # type: ignore
                               tokens: Dict[str, torch.LongTensor],
                               sequence_type="n/a") -> torch.Tensor:
        if self.bert_model.base_model_prefix == 'distilbert' or self.bert_model.base_model_prefix == 'electra':
            vectors = self.bert_model(**tokens)[0][:, 0, :]
        else:
            _, vectors = self.bert_model(**tokens, return_dict=False)
        return vectors
