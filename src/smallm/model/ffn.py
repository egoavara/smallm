"""SwiGLU Feed-Forward Network implementation.

SwiGLU is an activation function that combines the Swish activation
with a gating mechanism. It's used in LLaMA models and provides
better performance than standard ReLU or GELU.

Formula: SwiGLU(x) = (x @ W_gate) * SiLU(x @ W_up) @ W_down

Reference: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    The SwiGLU FFN consists of:
    1. Gate projection: x -> hidden (with SiLU activation)
    2. Up projection: x -> hidden (linear)
    3. Element-wise multiplication of gate and up
    4. Down projection: hidden -> output
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize SwiGLU FFN.

        Args:
            config: Model configuration
        """
        super().__init__()

        # Gate projection with SiLU activation
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)

        # Up projection (linear)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=False)

        # Down projection
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SwiGLU FFN.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # SwiGLU: SiLU(gate) * up
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up

        # Down projection
        output = self.w_down(hidden)
        output = self.dropout(output)

        return output
