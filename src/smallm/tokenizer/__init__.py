"""BPE Tokenizer implementation."""

from .base import Tokenizer
from .bpe_optimized import OptimizedBPE

# Rust-powered BPE (optional, much faster)
try:
    from .bpe_rust import RustBPE, RUST_AVAILABLE
except ImportError:
    RustBPE = None
    RUST_AVAILABLE = False

__all__ = ["Tokenizer", "OptimizedBPE", "RustBPE", "RUST_AVAILABLE"]
