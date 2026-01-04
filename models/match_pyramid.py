from typing import Dict, Iterator, List,Tuple
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MatchPyramid(nn.Module):
    @staticmethod
    def from_config(config):
        return MatchPyramid(config)

    def __init__(self, config):
        super(MatchPyramid, self).__init__()
        self.config = config
        self.q_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_q_emb'])
        if self.config['q_d_emb_same']:
            self.d_embed = self.q_embed
        else:
            self.d_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_d_emb'])

        self.cosine_module = CosineMatrixAttention()

        conv_output_size = self.config['conv_output_size']
        conv_kernel_size = self.config['conv_kernel_size']
        adaptive_pooling_size = self.config['adaptive_pooling_size']
        if len(conv_output_size) != len(conv_kernel_size) or len(conv_output_size) != len(adaptive_pooling_size):
            raise Exception("conv_output_size, conv_kernel_size, adaptive_pooling_size must have the same length")

        conv_layer_dict = OrderedDict()
        last_channel_out = 1
        for i in range(len(conv_output_size)):
            conv_layer_dict["pad " +str(i)] = nn.ConstantPad2d((0, conv_kernel_size[i][0] - 1,0, conv_kernel_size[i][1] - 1), 0)
            conv_layer_dict["conv "+str(i)] = nn.Conv2d(kernel_size=conv_kernel_size[i], in_channels=last_channel_out, out_channels=conv_output_size[i])
            conv_layer_dict["relu "+str(i)] = nn.ReLU()
            conv_layer_dict["pool "+str(i)] = nn.AdaptiveMaxPool2d(adaptive_pooling_size[i])
            last_channel_out = conv_output_size[i]

        self.conv_layers = nn.Sequential(conv_layer_dict)

        self.dense = nn.Linear(conv_output_size[-1] * adaptive_pooling_size[-1][0] * adaptive_pooling_size[-1][1], out_features=100, bias=True)
        self.dense2 = nn.Linear(100, out_features=10, bias=True)
        self.dense3 = nn.Linear(10, out_features=1, bias=False)

    def forward(self, q, d):
        # query|doc to query embedding
        query, q_mask = q['input_ids'], q['attention_mask']
        doc, d_mask = d['input_ids'], d['attention_mask']
        q_emb = self.q_embed(query)
        t_emb = self.d_embed(doc)

        cosine_matrix = self.cosine_module.forward(q_emb, t_emb)
        cosine_matrix = cosine_matrix[:,None,:,:]

        conv_result = self.conv_layers(cosine_matrix)

        conv_result_flat = conv_result.view(conv_result.size(0), -1)
        dense_out = F.relu(self.dense(conv_result_flat))
        dense_out = F.relu(self.dense2(dense_out))
        dense_out = self.dense3(dense_out)
        output = torch.squeeze(dense_out, 1)
        return output

    def get_param_stats(self):
        return "MP: / "
    def get_param_secondary(self):
        return {}
