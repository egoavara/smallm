"""Model configuration."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the LLaMA-style model.

    Attributes:
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads (query heads)
        n_kv_heads: Number of key-value heads (for GQA)
        d_model: Model dimension (embedding size)
        d_ff: Feed-forward network hidden dimension
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        rope_theta: Base for RoPE frequency computation
        norm_eps: Epsilon for RMSNorm
        dropout: Dropout probability
    """

    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    d_model: int = 512
    d_ff: int = 1376  # 2/3 * 4 * d_model for SwiGLU
    vocab_size: int = 32000
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.n_heads // self.n_kv_heads

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"


# Preset configurations
CONFIGS = {
    "tiny": ModelConfig(
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_model=256,
        d_ff=688,
        vocab_size=32000,
        max_seq_len=512,
    ),
    "small": ModelConfig(
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        d_model=512,
        d_ff=1376,
        vocab_size=32000,
        max_seq_len=512,
    ),
    "medium": ModelConfig(
        n_layers=12,
        n_heads=12,
        n_kv_heads=4,
        d_model=768,
        d_ff=2048,
        vocab_size=32000,
        max_seq_len=1024,
    ),
}
