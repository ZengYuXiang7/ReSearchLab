# coding : utf-8
# Author : Yuxiang Zeng
import torch
from torch.nn import *

class NTC(torch.nn.Module):
    def __init__(self, enc_in, config):
        super(NTC, self).__init__()
        self.config = config
        self.rank = config.rank
        self.window = config.window
        self.lstm = LSTM(self.rank, self.rank, batch_first=False)
        self.rainbow = torch.arange(-self.window + 1, 1).reshape(1, -1).to(self.config.device)
        self.attn = Sequential(Linear(2 * self.rank, 1), Tanh())
        self.user_embeds = Embedding(config.num_nodes, self.rank)
        self.item_embeds = Embedding(config.num_nodes, self.rank)
        self.time_embeds = Embedding(config.num_slots, self.rank)
        self.user_linear = Linear(self.rank, self.rank)
        self.item_linear = Linear(self.rank, self.rank)
        self.time_linear = Linear(self.rank, self.rank)

    def to_seq_id(self, tids):
        tids = tids.reshape(-1, 1).repeat(1, self.window)
        tids += self.rainbow
        tids = tids.relu().permute(1, 0)
        return tids

    def forward(self, user, item, time):
        user_embeds = self.user_embeds(user)
        item_embeds = self.item_embeds(item)
        time_embeds = self.time_embeds(self.to_seq_id(time))

        outputs, (hs, cs) = self.lstm.forward(time_embeds)

        # Attention [seq_len, batch, dim] -> [seq_len, batch, 1]
        hss = hs.repeat(self.window, 1, 1)
        attn = self.attn(torch.cat([outputs, hss], dim=-1))
        time_embeds = torch.sum(attn * outputs, dim=0)

        user_embeds = self.user_linear(user_embeds)
        item_embeds = self.item_linear(item_embeds)
        time_embeds = self.time_linear(time_embeds)

        raw_score = torch.sum(user_embeds * item_embeds * time_embeds, dim=-1)
        y = raw_score.sigmoid()
        return y.unsqueeze(-1)
