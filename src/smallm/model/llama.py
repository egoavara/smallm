"""LLaMA-style Language Model implementation.

This is a minimal implementation of a LLaMA-style decoder-only transformer
for educational purposes. It includes:
- RMSNorm (instead of LayerNorm)
- RoPE (Rotary Position Embeddings)
- GQA (Grouped Query Attention)
- SwiGLU (activation function)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional

from .config import ModelConfig
from .norm import RMSNorm
from .rope import precompute_freqs_cis
from .block import TransformerBlock


class LLaMA(nn.Module):
    """LLaMA-style Language Model.

    Architecture:
    1. Token Embedding (no position embedding - RoPE is used instead)
    2. N Transformer Blocks
    3. Final RMSNorm
    4. Output projection (tied with embedding weights)
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialize LLaMA model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Dropout after embedding
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # Output projection (language model head)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with output projection
        self.tok_emb.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        # Precompute causal mask
        self.register_buffer(
            "causal_mask",
            self._create_causal_mask(config.max_seq_len),
            persistent=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using standard deviation of 0.02."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _create_causal_mask(self, max_seq_len: int) -> torch.Tensor:
        """Create causal attention mask.

        Args:
            max_seq_len: Maximum sequence length

        Returns:
            Causal mask tensor
        """
        mask = torch.full((max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: Input token IDs of shape (batch, seq_len)
            targets: Optional target token IDs of shape (batch, seq_len)

        Returns:
            Tuple of (logits, loss):
            - logits: Output logits of shape (batch, seq_len, vocab_size)
            - loss: Cross-entropy loss if targets provided, else None
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}"

        # Token embedding
        h = self.tok_emb(input_ids)
        h = self.dropout(h)

        # Get RoPE frequencies for this sequence length
        freqs_cis = self.freqs_cis[:seq_len]

        # Get causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]

        # Apply transformer blocks
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = checkpoint(layer, h, freqs_cis, mask, use_reentrant=False)
            else:
                h = layer(h, freqs_cis, mask)

        # Final normalization
        h = self.norm(h)

        # Output projection
        logits = self.lm_head(h)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,  # Ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            input_ids: Starting token IDs of shape (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (1.0 = no change)
            top_k: If set, only sample from top k tokens
            top_p: If set, sample from smallest set with cumulative prob >= top_p

        Returns:
            Generated token IDs of shape (batch, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids
            if input_ids.shape[1] > self.config.max_seq_len:
                idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing.

        When enabled, activations are recomputed during backward pass
        instead of being stored, significantly reducing memory usage
        at the cost of ~20-30% slower training.

        Args:
            enable: Whether to enable gradient checkpointing
        """
        self.gradient_checkpointing = enable
