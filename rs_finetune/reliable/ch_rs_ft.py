"""#15 Channel-Token Hierarchical Randomized Smoothing Fine-Tuning.

Two halves:

- :func:`smooth_channel_tokens` — adds Gaussian noise to a Bernoulli-
  selected fraction of channel tokens during training. No-op at eval.
- :func:`mc_smooth_predict` — eval-time Monte Carlo aggregation: forward
  the model many times under noise, return the majority-vote class and a
  vote-count confidence statistic (placeholder for proper Cohen-style
  certificates added later).
"""

import torch


def smooth_channel_tokens(
    tokens: torch.Tensor,
    sigma: float,
    p_smooth: float,
    training: bool,
) -> torch.Tensor:
    """Add Gaussian-σ noise to a Bernoulli-p_smooth subset of channel tokens.

    Args:
        tokens: ``(B, C, D)`` channel-token features.
        sigma: noise standard deviation.
        p_smooth: probability per channel of being noised.
        training: only perturb when True.

    Returns the same shape as ``tokens``. Pure passthrough when
    ``training is False``, ``sigma == 0``, or ``p_smooth == 0``.
    """
    if not training or sigma == 0.0 or p_smooth == 0.0:
        return tokens
    if not 0.0 < p_smooth <= 1.0:
        raise ValueError(f"p_smooth must be in (0, 1], got {p_smooth}")
    B, C, _D = tokens.shape
    mask = (torch.rand(B, C, device=tokens.device) < p_smooth).to(tokens.dtype)
    noise = torch.randn_like(tokens) * sigma
    return tokens + mask.unsqueeze(-1) * noise
