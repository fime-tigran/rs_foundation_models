"""#20 LoRA-Null initialization.

Returns an orthonormal basis of a rank-``null_rank`` direction that is
orthogonal to the dominant principal directions of a sample of subset-
forward activations. Used to initialise ``LoRALayer.A`` so that, at the
start of training, ``A @ x ≈ 0`` for any ``x`` in the span of observed
subset activations — the adapter has zero initial effect on the subset
forward.
"""

import torch

from reliable.lora_layer import LoRALayer


def compute_activation_null_basis(
    activations: torch.Tensor, null_rank: int
) -> torch.Tensor:
    if activations.ndim != 2:
        raise ValueError(
            f"activations must be 2D (N, D), got shape {tuple(activations.shape)}"
        )
    _N, D = activations.shape
    if null_rank < 0 or null_rank > D:
        raise ValueError(
            f"null_rank must be in [0, {D}], got {null_rank}"
        )
    _U, _S, Vh = torch.linalg.svd(activations, full_matrices=True)
    V = Vh.T
    return V[:, D - null_rank :].contiguous()


def init_lora_a_in_null_space(
    lora: LoRALayer, activations: torch.Tensor, null_rank: int
) -> None:
    """Initialise ``lora.A`` so that ``A @ x ≈ 0`` for ``x`` in the span
    of ``activations``. LoRA ``rank`` must equal ``null_rank``.
    """
    if lora.A.shape[0] != null_rank:
        raise ValueError(
            f"LoRA rank ({lora.A.shape[0]}) must equal null_rank ({null_rank})"
        )
    U_null = compute_activation_null_basis(activations, null_rank=null_rank)
    with torch.no_grad():
        lora.A.copy_(U_null.T)  # (null_rank, D)
