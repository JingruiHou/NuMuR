# encoding=utf-8
import numpy as np
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class DUET(nn.Module):

    @staticmethod
    def from_config(cfg):
        return DUET(cfg)

    def __init__(self, config):
        super(DUET, self).__init__()
        self.config = config
        self.ARCH_TYPE = 2
        self.q_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_q_emb'])
        if self.config['q_d_emb_same']:
            self.d_embed = self.q_embed
        else:
            self.d_embed = nn.Embedding.from_pretrained(torch.tensor(config['embeddings']), freeze=config['freeze_d_emb'])

        max_query_terms = self.config['query_max_len']
        max_doc_terms = self.config['target_max_len']

        num_hidden_nodes = self.config['NUM_HIDDEN_NODES']
        term_window_size = self.config['TERM_WINDOW_SIZE']
        dropout_rate = self.config['DROPOUT_RATE']
        pooling_kernel_width_doc = self.config['POOLING_KERNEL_WIDTH_DOC']
        pooling_kernel_width_query = max_query_terms - term_window_size + 1  # 20 - 3 + 1 = 18
        num_pooling_windows_doc = (max_doc_terms - term_window_size + 1) - pooling_kernel_width_doc + 1  # (200 - 3 + 1) - 100 + 1 = 99

        self.duet_local = nn.Sequential(nn.Conv1d(max_doc_terms, num_hidden_nodes, kernel_size=1),
                                        nn.ReLU(),
                                        Flatten(),
                                        nn.Dropout(p=dropout_rate),
                                        nn.Linear(num_hidden_nodes * max_query_terms, num_hidden_nodes),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_rate),
                                        nn.Linear(num_hidden_nodes, num_hidden_nodes),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout_rate))
        self.duet_dist_q = nn.Sequential(nn.Conv1d(num_hidden_nodes, num_hidden_nodes, kernel_size=3),
                                         nn.ReLU(),
                                         nn.MaxPool1d(pooling_kernel_width_query),
                                         Flatten(),
                                         nn.Linear(num_hidden_nodes, num_hidden_nodes),
                                         nn.ReLU()
                                         )
        self.duet_dist_d = nn.Sequential(nn.Conv1d(num_hidden_nodes, num_hidden_nodes, kernel_size=3),
                                         nn.ReLU(),
                                         nn.MaxPool1d(pooling_kernel_width_doc, stride=1),
                                         nn.Conv1d(num_hidden_nodes, num_hidden_nodes, kernel_size=1),
                                         nn.ReLU()
                                         )
        self.duet_dist = nn.Sequential(Flatten(),
                                       nn.Dropout(p=dropout_rate),
                                       nn.Linear(num_hidden_nodes * num_pooling_windows_doc, num_hidden_nodes),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate),
                                       nn.Linear(num_hidden_nodes, num_hidden_nodes),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate))
        self.duet_comb = nn.Sequential(nn.Linear(num_hidden_nodes, num_hidden_nodes),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate),
                                       nn.Linear(num_hidden_nodes, num_hidden_nodes),
                                       nn.ReLU(),
                                       nn.Dropout(p=dropout_rate),
                                       nn.Linear(num_hidden_nodes, 1),
                                       nn.ReLU())
        self.scale = torch.tensor([0.1], requires_grad=False).to(self.config['device'])

    def forward(self, q, d):
        x_dist_q, x_mask_q = q['input_ids'], q['attention_mask']
        x_dist_d, x_mask_d = d['input_ids'], d['attention_mask']
        x_local = d['local']
        x_mask_q = x_mask_q.view(x_mask_q.size()[0], x_mask_q.size()[1], 1)
        x_mask_d = x_mask_d.view(x_mask_d.size()[0], x_mask_d.size()[1], 1)
        if self.ARCH_TYPE != 1:
            h_local = self.duet_local(x_local)
        if self.ARCH_TYPE > 0:
            h_dist_q = self.duet_dist_q((self.q_embed(x_dist_q) * x_mask_q).permute(0, 2, 1))
            h_dist_d = self.duet_dist_d((self.d_embed(x_dist_d) * x_mask_d).permute(0, 2, 1))
            h_dist = self.duet_dist(h_dist_q.unsqueeze(-1) * h_dist_d)
        y_score = self.duet_comb(
            (h_local + h_dist) if self.ARCH_TYPE == 2 else (h_dist if self.ARCH_TYPE == 1 else h_local)) * self.scale
        return torch.squeeze(y_score, 1)


if __name__ == '__main__':
    pass
