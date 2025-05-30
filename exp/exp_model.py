# coding : utf-8
# Author : Yuxiang Zeng
# 每次开展新实验都改一下这里
from exp.exp_base import BasicModel
from modules.TS.ts import TimeSeriesModel
from modules.ntc import NTC


class Model(BasicModel):
    def __init__(self, datamodule, config):
        super().__init__(config)
        self.config = config
        # self.input_size = datamodule.train_loader.dataset.x.shape[-1]
        self.input_size = 1
        self.hidden_size = config.rank

        if config.task == 'NTC':
            self.model = NTC(self.input_size, config)

        elif config.task == 'TS':
            self.model = TimeSeriesModel(self.input_size, config)

        elif config.task == 'AD':
            self.model = TimeSeriesModel(self.input_size, config)

        elif config.task == 'Recsys':
            self.model = TimeSeriesModel(self.input_size, config)

        else:
            raise ValueError(f"Unsupported model type: {config.model}")


