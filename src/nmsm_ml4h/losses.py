from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    covariates: float


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    valid = mask.to(torch.bool)
    use = valid.to(values.dtype)
    if weights is not None:
        use = use * weights.to(values.dtype)


    safe_values = torch.where(valid, values, torch.zeros_like(values))
    return (safe_values * use).sum() / use.sum().clamp_min(1.0)


def compute_loss(
    outputs: dict[str, torch.Tensor | None],
    batch: dict,
    allowed: torch.Tensor,
    weights: LossWeights,
) -> dict[str, torch.Tensor]:
    state_logits = outputs["state_logits"]
    location = outputs["covariate_location"]
    log_scale = outputs["covariate_log_scale"]
    observation_logits = outputs["observation_logits"]
    binary_logits = outputs["binary_logits"]
    assert state_logits is not None
    assert location is not None and log_scale is not None
    assert observation_logits is not None and binary_logits is not None
    transition_mask = allowed[batch["input_states"].clamp(0, allowed.shape[0] - 1)]
    masked_logits = state_logits.masked_fill(~transition_mask, -1e9)
    state_ce = F.cross_entropy(
        masked_logits.transpose(1, 2), batch["target_states"], ignore_index=-1, reduction="none"
    )
    state_loss = masked_mean(state_ce, batch["next_mask"])
    standardized = (
        batch["target_continuous_values"] - location
    ) * torch.exp(-log_scale)
    gaussian_nll = 0.5 * standardized.square() + log_scale + 0.5 * math.log(2.0 * math.pi)
    value_loss = masked_mean(gaussian_nll, batch["continuous_value_mask"])
    observation_nll = F.binary_cross_entropy_with_logits(
        observation_logits, batch["target_observed"], reduction="none"
    )
    observation_loss = masked_mean(
        observation_nll, batch["observation_loss_mask"]
    )
    binary_nll = F.binary_cross_entropy_with_logits(
        binary_logits, batch["target_binary_values"], reduction="none"
    )
    binary_loss = masked_mean(binary_nll, batch["binary_loss_mask"])
    cov_loss = value_loss + observation_loss + binary_loss
    total = state_loss + weights.covariates * cov_loss
    parts = {
        "state": state_loss,
        "covariates": cov_loss,
        "covariate_values": value_loss,
        "observations": observation_loss,
        "binary_covariates": binary_loss,
    }
    parts["total"] = total
    return parts
