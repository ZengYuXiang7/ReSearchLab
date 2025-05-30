# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import pandas as pd
from utils.data_scaler import get_scaler


def load_dataset(config):
    data = np.load('./datasets/NTC/abilene.npy')[:config.num_slots]
    thsh = np.percentile(data, 99)
    data[data > thsh] = thsh
    data /= thsh
    return data.astype(np.float32)
    #
    # if config.dataset == 'geant':
    #     data = np.load('./datasets/NTC/geant.npy')[:config.num_slots]
    #     thsh = np.percentile(data, config.quantile)
    #     data[data > thsh] = thsh
    #     data /= thsh
    #     return data.astype(np.float32)
    #
    # if config.dataset == 'harvard226':
    #     data = np.load('./datasets/NTC/harvard226.npy')[:config.num_slots]
    #     thsh = np.percentile(data, config.quantile)
    #     data[data > thsh] = thsh
    #     data /= thsh
    #     return data.astype(np.float32)
    #
    # if config.dataset == 'wsdream':
    #     data = np.load('./datasets/NTC/wsdream.npy')[:config.num_slots]
    #     thsh = np.percentile(data, config.quantile)
    #     data[data > thsh] = thsh
    #     data /= thsh
    #     return data.astype(np.float32)

    # return None


def get_tensor(dataset, config):
    df = load_dataset(config)
    print(df.shape)
    x = np.array(np.nonzero(df)).T  # shape: [n, 3]
    y = df[x[:, 0], x[:, 1], x[:, 2]].reshape(-1, 1)

    # 根据训练集对input进行特征归一化
    scaler = get_scaler(y, config)
    y = scaler.transform(y)
    # x = scaler.transform(x)

    x = x.astype(np.int32)
    y = y.astype(np.float32)
    return x, y, scaler