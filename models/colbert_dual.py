from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from typing import Dict
import torch


class ColBERTConfig(PretrainedConfig):
    model_type = "ColBERT"
    bert_model: str
    compression_dim: int = 768
    dropout: float = 0.0
    return_vecs: bool = False
    trainable: bool = True


class ColBERT(PreTrainedModel):
    """
    ColBERT model from: https://arxiv.org/pdf/2004.12832.pdf
    We use a dot-product instead of cosine per term (slightly better)
    """

    @staticmethod
    def from_config(config):
        # colbert_compression_dim: 768
        cfg = ColBERTConfig()
        cfg.bert_model = config["bert_pretrained_model"]
        cfg.compression_dim = config["colbert_compression_dim"]
        cfg.return_vecs = config.get("in_batch_negatives", False)
        cfg.trainable = config["bert_trainable"]
        cfg.dual_encoder = config["dual_encoder"]
        return ColBERT(cfg)

    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)
        self.bert_model_q = AutoModel.from_pretrained(cfg.bert_model)
        self.compressor_q = torch.nn.Linear(self.bert_model_q.config.hidden_size, cfg.compression_dim)
        for p in self.bert_model_q.parameters():
            p.requires_grad = cfg.trainable

        if cfg.dual_encoder:
            self.bert_model_d = AutoModel.from_pretrained(cfg.bert_model)
            for p in self.bert_model_d.parameters():
                p.requires_grad = cfg.trainable
            self.compressor_d = torch.nn.Linear(self.bert_model_d.config.hidden_size, cfg.compression_dim)
        else:
            self.bert_model_d = self.bert_model_q
            self.compressor_d = self.compressor_q

    def forward(self,
                query: Dict[str, torch.LongTensor],
                document: Dict[str, torch.LongTensor]):

        query_vecs, document_vecs = self.forward_representation(query, document)
        score = self.forward_aggregation(query_vecs, document_vecs, query["attention_mask"], document["attention_mask"])
        return score

    def forward_representation(self, q_tokens, d_tokens):
        q_rep = self.bert_model_q(**q_tokens)[0]  # assuming a distilbert model here
        q_vecs = self.compressor_q(q_rep)
        d_rep = self.bert_model_d(**d_tokens)[0]  # assuming a distilbert model here
        d_vecs = self.compressor_d(d_rep)
        return q_vecs, d_vecs

    def forward_aggregation(self, query_vecs, document_vecs, query_mask, document_mask):
        # create initial term-x-term scores (dot-product)
        score = torch.bmm(query_vecs, document_vecs.transpose(2, 1))
        # mask out padding on the doc dimension (mask by -1000, because max should not select those, setting it to 0 might select them)
        exp_mask = document_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~exp_mask] = - 10000
        # max pooling over document dimension
        score = score.max(-1).values
        # mask out paddding query values
        score[~(query_mask.bool())] = 0
        # sum over query values
        score = score.sum(-1)
        return score
