"""nanoGPT-style transformer block (trimmed for the visual-ml-model example).

Adapted in spirit from Andrej Karpathy's nanoGPT (MIT). This is a faithful
pre-norm GPT decoder block: causal self-attention (fused qkv) + GELU MLP with
two residual connections. The tool reads this source statically; it is never
executed by the MVP pipeline.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # fused query, key, value projections for all heads, in one matmul
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection back into the residual stream
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # causal mask, not a learned parameter, and not saved in the state_dict
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )
