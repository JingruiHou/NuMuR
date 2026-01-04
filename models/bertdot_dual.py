from typing import Dict, Union
import torch
from transformers import AutoModel
from transformers import PreTrainedModel, PretrainedConfig


class BERT_Dot_Config(PretrainedConfig):
    model_type = "BERT_Dot"
    bert_model: str
    trainable: bool = True
    compression_dim: int = -1  # if -1 add no compression layer, otherwise add 1 single linear layer (from bert_out_dim to compress_dim)
    return_vecs: bool = False  # whether to return the vectors in the training forward pass (for in-batch negative loss)


class BERTdot(PreTrainedModel):
    """
    The main dense retrieval model;
    this model does not concat query and document, rather it encodes them sep. and uses a dot-product between the two cls vectors
    """

    config_class = BERT_Dot_Config
    base_model_prefix = "bert_model"

    @staticmethod
    def from_config(config):
        cfg = BERT_Dot_Config()
        cfg.bert_model = config["bert_pretrained_model"]
        cfg.trainable = config["bert_trainable"]
        cfg.return_vecs = config.get("in_batch_negatives", False)
        cfg.compression_dim = config.get("bert_dot_compress_dim", -1)
        cfg.dual_encoder = config["dual_encoder"]
        return BERTdot(cfg)

    def __init__(self,
                 cfg) -> None:

        super().__init__(cfg)

        self.bert_model_q = AutoModel.from_pretrained(cfg.bert_model)
        self.compressor_q = torch.nn.Linear(self.bert_model_q.config.hidden_size, cfg.compression_dim)
        for p in self.bert_model_q.parameters():
            p.requires_grad = cfg.trainable
        if cfg.dual_encoder:
            self.bert_model_d = AutoModel.from_pretrained(cfg.bert_model)
            self.compressor_d = torch.nn.Linear(self.bert_model_d.config.hidden_size, cfg.compression_dim)
            for p in self.bert_model_d.parameters():
                p.requires_grad = cfg.trainable
        else:
            self.bert_model_d = self.bert_model_q
            self.compressor_d = self.compressor_q

    def forward(self, query: Dict[str, torch.LongTensor], document: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        query_vecs, document_vecs = self.forward_representation(query, document)
        score = torch.bmm(query_vecs.unsqueeze(dim=1), document_vecs.unsqueeze(dim=2)).squeeze(-1).squeeze(-1)
        return score

    def forward_representation(self,  q_tokens, d_tokens):
        q_rep = self.bert_model_q(**q_tokens)[0][:, 0, :]  # assuming a distilbert model here
        q_vecs = self.compressor_q(q_rep)
        d_rep = self.bert_model_d(**d_tokens)[0][:, 0, :] # assuming a distilbert model here
        d_vecs = self.compressor_d(d_rep)
        return q_vecs, d_vecs
