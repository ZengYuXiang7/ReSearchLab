# coding : utf-8
# Author : yuxiang Zeng

from default_config import *
from dataclasses import dataclass

@dataclass
class OtherConfig:
    classification: bool = False
    ablation: int = 0
    try_exp: int = -1


@dataclass
class TimeSeriesConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    task: str = 'TS'
    model: str = 'TS'
    dataset: str = 'TS'  # financial  TS
    seq_len: int = 96
    pred_len: int = 96
    ts_var: int = 0

    bs: int = 32
    rank: int = 56
    epochs: int = 200
    loss_func: str = 'MSELoss'  # L1Loss  MSELoss
    patience: int = 45
    multi_dataset: bool = True

    # 组件专区
    num_layers: int = 2
    norm_method: str = 'rms'
    ffn_method: str = 'ffn'
    att_method: str = 'self'
    dis_method: str = 'cosine'
    fft: bool = False
    revin: bool = False
    idx: int = 0


@dataclass
class NTCConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    task: str = 'NTC'
    model: str = 'NTC'
    dataset: str = 'NTC'
    density: float = 0.10

    bs: int = 256
    rank: int = 40
    epochs: int = 100
    loss_func: str = 'MSELoss'  # L1Loss  MSELoss
    patience: int = 10
    verbose: int = 1
    shuffle: bool = True
    scaler_method: str = 'minmax'

    # 组件专区
    num_nodes: int = 12
    num_slots: int = 48000
    window: int = 48
    idx: int = 0


@dataclass
class ADConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'NTC'
    dataset: str = 'NTC'
    density: float = 0.10

    bs: int = 256
    rank: int = 40
    epochs: int = 100
    loss_func: str = 'CrossEntropyLoss'  # # L1Loss  MSELoss
    patience: int = 10
    verbose: int = 1
    shuffle: bool = True
    scaler_method: str = 'minmax'

    # 组件专区
    num_nodes: int = 12
    num_slots: int = 48000
    window: int = 48
    idx: int = 0



@dataclass
class RecsysConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'NTC'
    dataset: str = 'NTC'
    density: float = 0.10

    bs: int = 256
    rank: int = 40
    epochs: int = 100
    loss_func: str = 'CrossEntropyLoss'  # L1Loss  MSELoss
    patience: int = 10
    verbose: int = 1
    shuffle: bool = True
    scaler_method: str = 'minmax'

    # 组件专区
    num_nodes: int = 12
    num_slots: int = 48000
    window: int = 48
    idx: int = 0