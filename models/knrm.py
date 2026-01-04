# encoding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import kernel_mu
from .model_utils import kernel_sigma


class KNRM(nn.Module):

    @staticmethod
    def from_config(cfg):
        return KNRM(cfg)

    def __init__(self, config):
        super(KNRM, self).__init__()
        self.config = config
        self.q_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_q_emb'])
        if self.config['q_d_emb_same']:
            self.d_embed = self.q_embed
        else:
            self.d_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_d_emb'])
        self.dense = nn.Linear(config['n_kernels'], 1)
        self.init_param()

    def init_param(self):
        self.mus = torch.FloatTensor(kernel_mu(self.config['n_kernels']))
        self.mus = self.mus.view(1, 1, 1, self.config['n_kernels']).to(self.config['device'])  # (1, 1, 1, n_kernels) view 操作是为了配合后面的 interaction matrix 的操作
        self.sigmas = torch.FloatTensor(kernel_sigma(self.config['n_kernels']))
        self.sigmas = self.sigmas.view(1, 1, 1, self.config['n_kernels']).to(self.config['device'])   # (1, 1, 1, n_kernels)

    def interaction_matrix(self, q_emb_norm, d_emb_norm, q_mask, t_mask):
        # translation matrix
        # match_matrix: (batch_size * query_length * doc_length * 1)
        match_matrix = torch.bmm(q_emb_norm, torch.transpose(d_emb_norm, 1, 2)).view(q_emb_norm.size()[0],
                                                                                     q_emb_norm.size()[1], d_emb_norm.size()[1], 1)
        # RBF Kernel layers
        # kernel_pooling: batch_size * query_length * doc_length * n_kernels
        kernel_pooling = torch.exp(-((match_matrix - self.mus) ** 2) / (2 * (self.sigmas ** 2)))
        # kernel_pooling_row: batch_size * query_length  * doc_length * n_kernels
        kernel_pooling_row = kernel_pooling * t_mask
        # pooling_row_sum -> batch_size * query_length * n_kernels
        pooling_row_sum = torch.sum(kernel_pooling_row, 2)
        # kernel_pooling -> batch_size * query_length * n_kernels
        log_pooling = torch.log(torch.clamp(pooling_row_sum, min=1e-10)) * q_mask * 0.01  # scale down the data
        # sum the value on col th: log_pooling_sum: (batch_size * n_kernels)
        log_pooling_sum = torch.sum(log_pooling, 1)
        return log_pooling_sum

    def forward(self, q, d):
        # query|doc to query embedding
        query, q_mask = q['input_ids'], q['attention_mask']
        doc, d_mask = d['input_ids'], d['attention_mask']
        q_emb = self.q_embed(query)
        t_emb = self.d_embed(doc)
        # normalize the q_emb| t_emb
        q_emb_norm = F.normalize(q_emb, p=2, dim=2)
        t_emb_norm = F.normalize(t_emb, p=2, dim=2)
        # reshape the mask the size
        q_mask = q_mask.view(q_mask.size()[0], q_mask.size()[1], 1)
        d_mask = d_mask.view(d_mask.size()[0], 1, d_mask.size()[1], 1)
        # build interation matrix
        log_sum_pooling = self.interaction_matrix(q_emb_norm, t_emb_norm, q_mask, d_mask)
        # connect the dense layers
        # output -> batch_size * 1
        # pair wise format
        # output = torch.tanh(self.dense(log_sum_pooling))
        # point wise format
        # output = F.softmax(self.dense(log_sum_pooling), 1)
        output = torch.sigmoid(self.dense(log_sum_pooling))
        # outptu -> batch_size
        output = torch.squeeze(output, 1)
        # return output
        return output


if __name__ == '__main__':
    pass
