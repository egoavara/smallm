"""Rotary Position Embeddings (RoPE) implementation.

RoPE encodes position information by rotating the query and key vectors
in the attention mechanism. This allows the model to capture relative
positions while maintaining compatibility with efficient attention implementations.

Reference: https://arxiv.org/abs/2104.09864
"""

import torch


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (cis).

    This computes the frequencies used in RoPE:
    freqs[i] = 1 / (theta^(2i/dim)) for i = 0, 1, ..., dim/2-1

    Args:
        dim: Dimension of the embeddings (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation
        device: Device to create tensor on

    Returns:
        Complex tensor of shape (max_seq_len, dim//2) containing cis values
    """
    # Compute frequencies: theta^(-2i/dim) for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Create position indices
    t = torch.arange(max_seq_len, device=device)

    # Outer product: (max_seq_len,) x (dim/2,) -> (max_seq_len, dim/2)
    freqs = torch.outer(t, freqs)

    # Convert to complex exponentials: e^(i*freq) = cos(freq) + i*sin(freq)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting with attention tensors.

    Args:
        freqs_cis: Frequency tensor of shape (seq_len, head_dim//2)
        x: Input tensor of shape (batch, seq_len, n_heads, head_dim)

    Returns:
        Reshaped frequency tensor for broadcasting
    """
    ndim = x.ndim
    assert ndim >= 2
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    # Shape: (1, seq_len, 1, head_dim//2)
    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1]

    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    The rotation is applied by:
    1. Viewing real tensors as complex
    2. Multiplying by complex exponentials (rotation)
    3. Converting back to real

    Args:
        xq: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        xk: Key tensor of shape (batch, seq_len, n_kv_heads, head_dim)
        freqs_cis: Precomputed frequencies of shape (seq_len, head_dim//2)

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # View real tensors as complex: (batch, seq_len, n_heads, head_dim//2, 2) -> complex
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs for broadcasting
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    # Apply rotation (complex multiplication)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
