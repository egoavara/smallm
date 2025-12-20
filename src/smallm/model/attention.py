"""Grouped Query Attention (GQA) implementation.

GQA is a memory-efficient attention variant where multiple query heads
share the same key-value heads. This reduces the KV cache size during
inference while maintaining model quality.

Reference: https://arxiv.org/abs/2305.13245
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig
from .rope import apply_rotary_emb


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with RoPE.

    In GQA, we have:
    - n_heads query heads
    - n_kv_heads key-value heads (n_kv_heads <= n_heads)
    - Each KV head is shared by (n_heads // n_kv_heads) query heads
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize GQA.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_kv_groups = config.n_kv_groups  # queries per kv head
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        # Query projection: d_model -> n_heads * head_dim
        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)

        # Key projection: d_model -> n_kv_heads * head_dim
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)

        # Value projection: d_model -> n_kv_heads * head_dim
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)

        # Output projection: n_heads * head_dim -> d_model
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of GQA.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            freqs_cis: RoPE frequencies of shape (seq_len, head_dim//2)
            mask: Optional attention mask of shape (seq_len, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V projections
        xq = self.wq(x)  # (batch, seq_len, n_heads * head_dim)
        xk = self.wk(x)  # (batch, seq_len, n_kv_heads * head_dim)
        xv = self.wv(x)  # (batch, seq_len, n_kv_heads * head_dim)

        # Reshape for multi-head attention
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # Expand KV heads to match query heads for GQA
        # (batch, seq_len, n_kv_heads, head_dim) -> (batch, seq_len, n_heads, head_dim)
        xk = self._repeat_kv(xk)
        xv = self._repeat_kv(xv)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Compute attention scores
        # (batch, n_heads, seq_len, head_dim) @ (batch, n_heads, head_dim, seq_len)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, xv)

        # Reshape back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        return self.wo(output)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads.

        Args:
            x: Tensor of shape (batch, seq_len, n_kv_heads, head_dim)

        Returns:
            Tensor of shape (batch, seq_len, n_heads, head_dim)
        """
        if self.n_kv_groups == 1:
            return x

        batch_size, seq_len, n_kv_heads, head_dim = x.shape

        # Expand: (batch, seq_len, n_kv_heads, 1, head_dim)
        x = x.unsqueeze(3)

        # Repeat: (batch, seq_len, n_kv_heads, n_kv_groups, head_dim)
        x = x.expand(batch_size, seq_len, n_kv_heads, self.n_kv_groups, head_dim)

        # Reshape: (batch, seq_len, n_heads, head_dim)
        return x.reshape(batch_size, seq_len, self.n_heads, head_dim)
