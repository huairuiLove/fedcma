"""Backbone modules ported from MultimodalMIMIC (Xinlu Zhang et al., ICML'23):
multiTimeAttention (irregular-time attention encoder) and the learned time
embedding (periodic + linear) used by both modality branches.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class multiTimeAttention(nn.Module):
    """Attention over irregular time points with learned time embeddings.

    query: time embeddings of a uniform query grid [B, Tq, embed_time]
    key:   time embeddings of observation times      [B, Tk, embed_time]
    value: observations                              [B, Tk, dim]
    -> [B, Tq, nhidden]
    """

    def __init__(self, input_dim, nhidden=16, embed_time=16, num_heads=1):
        super().__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim * num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim=-2)
        if dropout is not None:
            p_attn = F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn * value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        batch, seq_len, dim = value.size()
        if mask is not None:
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class TimeEmbed(nn.Module):
    """learn_time_embedding from the backbone: [linear | sin(periodic)]."""

    def __init__(self, embed_time):
        super().__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.embed_time = embed_time

    def forward(self, tt):
        out2 = torch.sin(self.periodic(tt.unsqueeze(-1)))
        out1 = self.linear(tt.unsqueeze(-1))
        return torch.cat([out1, out2], -1)


class TransformerEncoder(nn.Module):
    """Light transformer encoder over a modality's time grid (backbone ts_mem/txt_mem)."""

    def __init__(self, embed_dim, num_heads, layers, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=4 * embed_dim,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x):
        return self.encoder(x)


class gateMLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, output_dim), nn.Sigmoid())
        for m in self.gate:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return self.gate(x)
