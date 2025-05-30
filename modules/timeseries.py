# coding : utf-8
# Author : Yuxiang Zeng
import torch

from layers.dft import DFT
from layers.encoder.position_enc import PositionEncoding
from layers.encoder.seq_enc import SeqEncoder
from layers.encoder.token_emc import TokenEmbedding
from layers.feedforward.moe import MoE
from layers.revin import RevIN
from layers.transformer import Transformer

class Backbone(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(Backbone, self).__init__()
        self.config = config
        self.rank = config.rank
        self.pred_len = config.pred_len
        self.seq_len = config.seq_len
        self.revin = config.revin
        self.fft = config.fft
        if self.revin:
            self.revin_layer = RevIN(num_features=enc_in, affine=False, subtract_last=False)

        if self.fft:
            self.seasonality_and_trend_decompose = DFT(2)
        self.projection = torch.nn.Linear(enc_in, config.rank, bias=True)
        self.predict_linear = torch.nn.Linear(config.seq_len, config.pred_len + config.seq_len)

        self.encoder = Transformer(
            self.rank,
            num_heads=4,
            num_layers=config.num_layers,
            norm_method=config.norm_method,
            ffn_method=config.ffn_method,
            att_method=config.att_method
        )

        self.decoder = torch.nn.Linear(config.rank, 1)


    def forward(self, x, x_mark):
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.fft:
            x = self.seasonality_and_trend_decompose(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        x_enc = self.projection(x)
        x_enc = self.decoder(x_enc)
        y = x_enc[:, -self.pred_len:, :].squeeze(-1)  # [B, L, D]

        # denorm
        if self.revin:
            y = self.revin_layer(y, 'denorm')
            y = y[:, :, -1]  # [B, L, 1]
        return y