from typing import Dict, Iterator, List,Tuple
from collections import OrderedDict

import torch
import torch.nn as nn


class MaskedSoftmax(nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)


class CosineMatrixAttention(nn.Module):
    """
    Computes attention between every entry in matrix_1 with every entry in matrix_2 using cosine
    similarity.
    Registered as a `MatrixAttention` with name "cosine".
    """
    @staticmethod
    def tiny_value_of_dtype(dtype: torch.dtype):
        """
         Reference code (paper author): https://github.com/allenai/allennlp/blob/main/allennlp/modules/matrix_attention/cosine_matrix_attention.py
        Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
        issues such as division by zero.
        This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
        Only supports floating point dtypes.
        """
        if not dtype.is_floating_point:
            raise TypeError("Only supports floating point dtypes.")
        if dtype == torch.float or dtype == torch.double:
            return 1e-13
        elif dtype == torch.half:
            return 1e-4
        else:
            raise TypeError("Does not support dtype " + str(dtype))

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        a_norm = matrix_1 / (
            matrix_1.norm(p=2, dim=-1, keepdim=True) + self.tiny_value_of_dtype(matrix_1.dtype)
        )
        b_norm = matrix_2 / (
            matrix_2.norm(p=2, dim=-1, keepdim=True) + self.tiny_value_of_dtype(matrix_2.dtype)
        )
        return torch.bmm(a_norm, b_norm.transpose(-1, -2))


class DRMM(nn.Module):

    def from_config(cfg):
        return DRMM(cfg)
    '''
    Paper: A Deep Relevance Matching Model for Ad-hoc Retrieval, Guo et al., CIKM'16
    only works with fixed word embeddings !
    '''
    def __init__(self, config):
        super(DRMM, self).__init__()
        self.config = config
        self.q_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_q_emb'])
        if self.config['q_d_emb_same']:
            self.d_embed = self.q_embed
        else:
            self.d_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_d_emb'])
        self.cosine_module = CosineMatrixAttention()
        self.bin_count = self.config['bin_count']
        self.embed_dim = config['embeddings'].shape[1]
        self.matching_classifier = nn.Sequential(nn.Linear(self.bin_count, self.bin_count),nn.Tanh(), nn.Linear(self.bin_count, 1), nn.Tanh())
        self.query_gate = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Tanh(), nn.Linear(self.embed_dim, 1), nn.Tanh())
        self.query_softmax = MaskedSoftmax()

    def forward(self, q: Dict[str, torch.Tensor], d: Dict[str, torch.Tensor]) -> torch.Tensor:
        query, q_mask = q['input_ids'], q['attention_mask']
        doc, d_mask = d['input_ids'], d['attention_mask']
        q_emb = self.q_embed(query) * q_mask.unsqueeze(-1)
        t_emb = self.d_embed(doc) * d_mask.unsqueeze(-1)
        cosine_matrix = self.cosine_module.forward(q_emb, t_emb).cpu()

        histogram_tensor = torch.empty((cosine_matrix.shape[0], cosine_matrix.shape[1], self.bin_count))
        for b in range(cosine_matrix.shape[0]):
            for q in range(cosine_matrix.shape[1]):
                histogram_tensor[b, q] = torch.histc(cosine_matrix[b, q], bins=self.bin_count, min=-1, max=1)

        histogram_tensor = histogram_tensor.to(self.config['device'])
        classified_matches_per_query = self.matching_classifier(torch.log1p(histogram_tensor))
        # ----------------------------------------------
        query_gates_raw = self.query_gate(q_emb)
        query_gates = self.query_softmax(query_gates_raw.squeeze(-1),q_mask).unsqueeze(-1)

        #
        # combine it all
        # ----------------------------------------------
        scores = torch.sum(classified_matches_per_query * query_gates, dim=1)
        output = torch.squeeze(scores, 1)
        output = torch.sigmoid(output)
        return output

    def get_param_stats(self):
        return "DRMM: -"


