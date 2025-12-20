"""Transformer Block implementation.

Each block consists of:
1. Pre-RMSNorm + Attention + Residual
2. Pre-RMSNorm + FFN + Residual

This is the "Pre-Norm" architecture used in LLaMA.
"""

import torch
import torch.nn as nn
from typing import Optional

from .config import ModelConfig
from .norm import RMSNorm
from .attention import GroupedQueryAttention
from .ffn import SwiGLU


class TransformerBlock(nn.Module):
    """A single transformer block with Pre-Norm architecture."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize transformer block.

        Args:
            config: Model configuration
        """
        super().__init__()

        # Pre-normalization for attention
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Grouped Query Attention
        self.attention = GroupedQueryAttention(config)

        # Pre-normalization for FFN
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # SwiGLU Feed-Forward Network
        self.ffn = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            freqs_cis: RoPE frequencies
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Attention with residual connection
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)

        # FFN with residual connection
        out = h + self.ffn(self.ffn_norm(h))

        return out
