'Exact GDirMixDTMC likelihood from Maystre et al. (UAI 2022).'

from __future__ import annotations

import torch


def generalized_dirichlet_mask(mask: torch.Tensor) -> torch.Tensor:
    tail = torch.flip(
        torch.cumsum(torch.flip(mask.to(torch.int64), dims=(1,)), dim=1),
        dims=(1,),
    ).bool()
    return mask[:, :-1] & tail[:, 1:]


def generalized_dirichlet_loglike(
    params: torch.Tensor,
    features: torch.Tensor,
    counts: torch.Tensor,
    mask: torch.Tensor,
    offsets: torch.Tensor | None = None,
    l2: float = 0.0,
) -> torch.Tensor:
    'Return the published generalized-Dirichlet compound log-likelihood.'
    if params.ndim != 4 or params.shape[-1] != 2:
        raise ValueError("params must have shape [D,N,N-1,2]")
    linear = torch.einsum("md,dijq->mijq", features, params)
    concentration = torch.exp(linear)
    alpha, beta = concentration[..., 0], concentration[..., 1]
    if offsets is not None:
        offset_aggregate = torch.flip(
            torch.cumsum(torch.flip(offsets, dims=(-1,)), dim=-1), dims=(-1,)
        )
        alpha = alpha + offsets[..., :-1]
        beta = beta + offset_aggregate[..., 1:]
    aggregate = torch.flip(
        torch.cumsum(torch.flip(counts, dims=(-1,)), dim=-1), dims=(-1,)
    )
    values = (
        torch.lgamma(alpha + counts[..., :-1])
        + torch.lgamma(beta + aggregate[..., 1:])
        + torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        - torch.lgamma(alpha + beta + aggregate[..., :-1])
    )
    active = generalized_dirichlet_mask(mask)
    likelihood = torch.where(
        active[None, ...], values, torch.zeros_like(values)
    ).sum()
    return likelihood - float(l2) * params.square().sum()
