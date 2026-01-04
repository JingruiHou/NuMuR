# encoding=utf-8
from tqdm import tqdm
import csv
import numpy as np
import sys
import csv
import torch

csv.field_size_limit(sys.maxsize)


def separate_parameters_by_type(model):
    # 为 Embedding 层的参数
    embedding_params = []
    # 为其他层的参数
    other_params = []
    for module in model.children():
        if isinstance(module, torch.nn.Embedding):
            embedding_params += list(module.parameters())
        else:
            other_params += list(module.parameters())

    return embedding_params, other_params


def build_idf_dic(dict_path):
    idfs = {}
    with open(dict_path, mode='r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            idfs[row[0]] = float(row[1])
    return idfs


def build_glove_embedding_dic(emb_path, vocab_path, emb_dim):
    vocab = {}
    with open(vocab_path, mode='r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            row = line.strip().split()
            vocab[row[0]] = int(row[1])
    VOCAB_SIZE = len(vocab) + 1
    embeddings = np.zeros((VOCAB_SIZE, emb_dim), dtype=np.float32)
    with open(emb_path, mode='r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            cols = line.split()
            idx = vocab.get(cols[0], 0)
            if idx > 0:
                for i in range(emb_dim):
                    embeddings[idx, i] = float(cols[i + 1])
    return vocab, embeddings


def kernel_mu(n_kernels):
    """
    get mu for each guassian kernel, Mu is the middele of each bin
    :param n_kernels: number of kernels( including exact match). first one is the exact match
    :return: mus, a list of mu
    """
    mus = [1]  # exact match
    if n_kernels == 1:
        return mus
    bin_step = (1 - (-1)) / (n_kernels - 1)  # score from [-1, 1]
    mus.append(1 - bin_step / 2)  # the margain mu value
    for k in range(1, n_kernels - 1):
        mus.append(mus[k] - bin_step)
    return mus


def kernel_sigma(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels(including the exact match)
    :return: sigmas, a list of sigma
    """
    sigmas = [0.001]  # exact match small variance means exact match ?
    if n_kernels == 1:
        return sigmas
    return sigmas + [0.1] * (n_kernels - 1)


if __name__ == '__main__':
    print(kernel_mu(11))
    print(kernel_sigma(11))
