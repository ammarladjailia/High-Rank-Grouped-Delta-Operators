import torch

@torch.no_grad()
def active_spectral_proxy(beta: torch.Tensor) -> torch.Tensor:
    return (1.0 - beta).abs().amax(dim=-1)

@torch.no_grad()
def active_contraction_margin(beta: torch.Tensor) -> torch.Tensor:
    return 1.0 - active_spectral_proxy(beta)

@torch.no_grad()
def full_operator_proxy_radius(
    beta: torch.Tensor,
    rank: int,
    group_dim: int,
) -> torch.Tensor:
    if rank < group_dim:
        return torch.ones_like(beta[..., 0])
    return active_spectral_proxy(beta)
