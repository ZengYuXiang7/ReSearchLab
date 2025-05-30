# coding : utf-8
# Author : yuxiang Zeng
# 根据场景需要来改这里的input形状
from torch.utils.data import Dataset
import numpy as np


class TensorDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        i, j, k = self.x[idx][1], self.x[idx][2], self.x[idx][0]
        value = self.y[idx].reshape(1)
        return i, j, k, value

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        i, j, k, value = zip(*batch)
        i, j, k = default_collate(i).long(), default_collate(j).long(), default_collate(k).long()
        value = default_collate(value)
        return i, j, k, value


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, mode, config):
        self.config = config
        self.x = x
        self.y = y
        self.mode = mode

    def __len__(self):
        # return len(self.x)
        return len(self.x) - self.config.seq_len - self.config.pred_len + 1

    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.config.seq_len
        r_begin = s_end
        r_end = r_begin + self.config.pred_len

        x = self.x[s_begin:s_end][:, -1]
        x_mark = self.x[s_begin:s_end][:, :-1]
        y = self.y[r_begin:r_end]
        return x, x_mark, y

    def custom_collate_fn(self, batch, config):
        from torch.utils.data.dataloader import default_collate
        x, x_mark, y = zip(*batch)
        x, y = default_collate(x), default_collate(y)
        x_mark = default_collate(x_mark)
        return x, x_mark, y