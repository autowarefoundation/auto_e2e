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
