import torch.nn as nn

from ...registry import BLOCKS
from ..base_module import BaseModule


@BLOCKS.register_class('transformer_layer')
class TransformerLayer(BaseModule):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, n_dims, n_heads, n_feedforward_dims=2048, layernorm=False, dropout=0.1, activation='relu', **kwargs):
        super().__init__()
        self.n_dims = n_dims
        self.n_heads = n_heads
        self.n_feedforward_dims = n_feedforward_dims
        self.layernorm = layernorm
        self.dropout = dropout
        assert activation in ('relu', 'gelu')
        self.activation = activation

        if layernorm:
            self.layer = nn.TransformerEncoderLayer(n_dims, n_heads, n_feedforward_dims, dropout=dropout, activation=activation, **kwargs)
        else:
            self.mha = nn.MultiheadAttention(embed_dim=n_dims, num_heads=n_heads, dropout=dropout)
            self.fc1 = nn.Linear(n_dims, n_feedforward_dims)
            self.fc2 = nn.Linear(n_feedforward_dims, n_dims)
            self.do = nn.Dropout(dropout)
            self.do1 = nn.Dropout(dropout)
            self.do2 = nn.Dropout(dropout)
            self.acti = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x):
        if self.layernorm:
            out = self.layer(x)
        else:
            out = self.do1(self.mha(x, x, x)[0]) + x

            out = self.do(self.acti(self.fc1(out)))
            out = self.fc2(out)
            out = self.do2(out) + x
        return out
