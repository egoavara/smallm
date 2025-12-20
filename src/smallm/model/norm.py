"""RMSNorm implementation.

RMSNorm (Root Mean Square Layer Normalization) is used in LLaMA models
instead of LayerNorm. It's more computationally efficient as it doesn't
compute the mean, only the RMS.

Reference: https://arxiv.org/abs/1910.07467
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes by the RMS of the input, without centering (no mean subtraction).
    This is computationally cheaper than LayerNorm while maintaining similar performance.

    Formula: output = x / RMS(x) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Input dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized and scaled tensor
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
