import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TSVTextDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            column_num = len(lines[0].strip().split('\t'))
            pbar = tqdm(total=len(lines), desc=f'reading dataset file:{file_path} ...')
            for line in lines:
                arr = line.strip().split('\t')
                assert len(arr) == column_num
                self.samples.append(arr)
                pbar.update(1)

    def __len__(self):
        # 数据集的大小是样本列表的长度
        return len(self.samples)

    def __getitem__(self, index):
        # 对于给定的索引，我们返回对应的样本
        return self.samples[index]

    def get_all_samples(self):
        return self.samples


class SimpleStringDataset(Dataset):
    def __init__(self, *arrays):
        # 假设所有的数组长度相同
        self.arrays = arrays

    def __len__(self):
        # 数据集的大小是样本列表的长度
        return len(self.arrays[0])

    def __getitem__(self, index):
        # 对于给定的索引，我们返回对应的所有数组的样本
        return tuple(array[index] for array in self.arrays)
