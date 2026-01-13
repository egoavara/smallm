"""SmallM Quantization Module.

Post-Training Quantization (PTQ) for SmallM models.
Supports FP16 and INT8 dynamic quantization.
"""

from .fp16 import convert_to_fp16, save_fp16_checkpoint
from .int8 import quantize_dynamic_int8, save_int8_checkpoint
from .loader import load_quantized_model
from .utils import get_model_size, get_model_size_str

__all__ = [
    # FP16
    "convert_to_fp16",
    "save_fp16_checkpoint",
    # INT8
    "quantize_dynamic_int8",
    "save_int8_checkpoint",
    # Loader
    "load_quantized_model",
    # Utils
    "get_model_size",
    "get_model_size_str",
]
