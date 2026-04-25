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
import torch.nn as nn


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


@torch.no_grad()
def mc_smooth_predict(
    model: nn.Module,
    tokens: torch.Tensor,
    n_mc: int,
    sigma: float,
    p_smooth: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monte-Carlo aggregate logits over ``n_mc`` smoothed forwards.

    Returns ``(majority_class, vote_count_for_majority)`` per batch
    element. ``vote_count_for_majority`` is the integer number of MC
    samples whose argmax agreed with the final majority — a simple
    confidence statistic. (A proper Clopper-Pearson lower bound for
    Cohen-style certified radii is left for Phase 7.)
    """
    if n_mc < 1:
        raise ValueError(f"n_mc must be >= 1, got {n_mc}")
    all_preds = []
    for _ in range(n_mc):
        smoothed = smooth_channel_tokens(
            tokens, sigma=sigma, p_smooth=p_smooth, training=True,
        )
        logits = model(smoothed)
        all_preds.append(logits.argmax(dim=-1))
    stacked = torch.stack(all_preds, dim=0)              # (n_mc, B)
    majority = torch.mode(stacked, dim=0).values         # (B,)
    vote_count = (stacked == majority.unsqueeze(0)).sum(dim=0)  # (B,)
    return majority, vote_count
