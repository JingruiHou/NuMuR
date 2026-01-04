# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import kernel_mu
from .model_utils import kernel_sigma



class CKNRM(nn.Module):
    @staticmethod
    def from_config(cfg):
        return CKNRM(cfg)

    def __init__(self, config):
        super(CKNRM, self).__init__()
        self.config = config
        self.q_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_q_emb'])
        if self.config['q_d_emb_same']:
            self.d_embed = self.q_embed
        else:
            self.d_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_d_emb'])
        self.conv_uni = nn.Sequential(nn.Conv2d(1, self.config['out_kernel'], (1, 300)), nn.ReLU())
        self.conv_bi = nn.Sequential(nn.Conv2d(1, self.config['out_kernel'], (2, 300)), nn.ReLU())
        self.conv_tri = nn.Sequential(nn.Conv2d(1, self.config['out_kernel'], (3, 300)), nn.ReLU())
        self.conv_grams = [self.conv_uni, self.conv_bi, self.conv_tri]
        self.dense = nn.Linear(self.config['n_kernels'] * self.config['max_grams'] ** 2, 1)
        self.get_param()

    def get_param(self):
        self.mus = torch.FloatTensor(kernel_mu(self.config['n_kernels']))
        self.sigmas = torch.FloatTensor(kernel_sigma(self.config['n_kernels']))
        self.mus = self.mus.view(self.config['n_kernels']).to(self.config['device'])
        self.sigmas = self.sigmas.view(self.config['n_kernels']).to(self.config['device'])

    def conv_and_norm(self, q_emb, d_emb, q_mask, t_mask):
        q_emb_conv = [torch.transpose(torch.squeeze(self.conv_grams[i](q_emb.unsqueeze(1)), dim=-1), 1, 2) for i in range(self.config['max_grams'])]
        t_emb_conv = [torch.squeeze(self.conv_grams[j](d_emb.unsqueeze(1)), dim=-1) for j in range(self.config['max_grams'])]
        q_emb_norm = [F.normalize(q_item, p=2, dim=2, eps=1e-10) for q_item in q_emb_conv]
        t_emb_norm = [F.normalize(t_item, p=2, dim=1, eps=1e-10) for t_item in t_emb_conv]
        q_masks = [q_mask[:, :q_mask.size()[1] - m].unsqueeze(2) for m in range(self.config['max_grams'])]
        t_masks = [t_mask[:, :t_mask.size()[1] - n].unsqueeze(1).unsqueeze(3) for n in range(self.config['max_grams'])]
        return q_emb_norm, t_emb_norm, q_masks, t_masks

    def interaction_matrix(self, q_emb_norm, d_emb_norm, q_mask, t_mask):
        match_matrix = torch.unsqueeze(torch.bmm(q_emb_norm, d_emb_norm), 3)
        kernel_pooling = torch.exp(-((match_matrix - self.mus) ** 2) / (2 * (self.sigmas ** 2)))
        kernel_row = kernel_pooling * t_mask
        pooling_row_sum = torch.sum(kernel_row, 2)
        log_pooling = torch.log(torch.clamp(pooling_row_sum, min=1e-10)) * q_mask * 0.01  # scale down the data
        log_pooling_sum = torch.sum(log_pooling, 1)
        return log_pooling_sum

    def forward(self, q, d):
        query, q_mask = q['input_ids'], q['attention_mask']
        doc, d_mask = d['input_ids'], d['attention_mask']
        q_emb = self.q_embed(query)
        t_emb = self.d_embed(doc)
        q_emb_norm, t_emb_norm, q_masks, d_masks = self.conv_and_norm(q_emb, t_emb, q_mask, d_mask)
        mm_total = []
        for q_ix in range(self.config['max_grams']):
            for t_ix in range(self.config['max_grams']):
                # pass  # d_mask 和 q_mask 的数据还要调整；
                mm_total.append(self.interaction_matrix(q_emb_norm[q_ix], t_emb_norm[t_ix], q_masks[q_ix], d_masks[t_ix]))
        log_pooling_sums = torch.cat(mm_total, 1)
        output = torch.sigmoid(self.dense(log_pooling_sums))
        output = torch.squeeze(output, dim=-1)
        return output


if __name__ == "__main__":
    pass
