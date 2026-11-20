import torch
from models.hrg_delta import HRGDeltaLayer

def test_hrg_delta_gradients():
    layer = HRGDeltaLayer(
        in_channels=32,
        rank=4,
        groups=4,
        dynamic_q=True,
        dynamic_beta=True,
        beta_max=2.0,
    )

    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    loss = layer(x).square().mean()
    loss.backward()

    assert x.grad is not None
    assert layer.q_gen.weight.grad is not None
    assert layer.beta_gen.weight.grad is not None

def test_qr_orthonormality():
    layer = HRGDeltaLayer(
        in_channels=32,
        rank=4,
        groups=4,
        dynamic_q=True,
        dynamic_beta=True,
    )
    x = torch.randn(2, 32, 8, 8)
    layer(x)
    assert layer.last_orthogonality_error < 1e-5

def test_gate_bounds():
    layer = HRGDeltaLayer(
        in_channels=32,
        rank=4,
        groups=4,
        dynamic_q=True,
        dynamic_beta=True,
        beta_max=2.0
    )
    x = torch.randn(2, 32, 8, 8)
    layer(x)
    assert layer.last_beta_max <= 2.0
    assert layer.last_beta.min() >= 0.0

def test_fixed_operator_preserves_orthogonal_complement():
    batch, channels, height, width = 2, 32, 8, 8
    groups, rank = 4, 4
    group_dim = channels // groups
    spatial = height * width
    
    layer = HRGDeltaLayer(
        in_channels=channels,
        rank=rank,
        groups=groups,
        dynamic_q=False,
        dynamic_beta=False,
        beta_max=2.0
    )
    
    # disable value map to make it A = I - Q B Q^T
    layer.value_map = torch.nn.Identity()
    
    x = torch.randn(batch, channels, height, width)
    
    # Get Q
    descriptor = layer.pool(x).flatten(1)
    q = layer._get_q(descriptor) # [B, G, D, R]
    
    # Create v_perp such that Q^T v_perp = 0
    v = torch.randn(batch, groups, group_dim, spatial)
    q_t_v = torch.einsum("bgdr,bgdp->bgrp", q, v)
    proj_v = torch.einsum("bgdr,bgrp->bgdp", q, q_t_v)
    v_perp = v - proj_v
    
    x_perp = v_perp.reshape(batch, channels, height, width)
    
    # The output of layer(x_perp) should be A(x_perp). Wait, A(x) = x + Q B Q^T (V(x) - x). 
    # Here V(x) = x because Identity, so V(x) - x = 0.
    # Ah, to test A = I - Q B Q^T, we want to simulate V(x) = 0.
    class ZeroMap(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)
    layer.value_map = ZeroMap()
    
    out_perp = layer(x_perp)
    
    assert torch.allclose(out_perp, x_perp, atol=1e-5)

def test_equivalence_of_forms():
    layer = HRGDeltaLayer(in_channels=32)
    x = torch.randn(2, 32, 8, 8)
    
    batch_size, channels, height, width = x.shape
    spatial_size = height * width
    descriptor = layer.pool(x).flatten(1)
    q = layer._get_q(descriptor)
    beta = layer._get_beta(descriptor)
    v_ambient = layer.value_map(x)
    
    x_grouped = x.reshape(batch_size, layer.groups, layer.group_dim, spatial_size)
    v_grouped = v_ambient.reshape(batch_size, layer.groups, layer.group_dim, spatial_size)
    
    # Form 1: Ambient
    res_amb = v_grouped - x_grouped
    res_sub = torch.einsum("bgdr,bgdp->bgrp", q, res_amb)
    gated_sub = beta.unsqueeze(-1) * res_sub
    upd_1 = torch.einsum("bgdr,bgrp->bgdp", q, gated_sub)
    
    # Form 2: Coordinate
    v_c = torch.einsum("bgdr,bgdp->bgrp", q, v_grouped)
    x_c = torch.einsum("bgdr,bgdp->bgrp", q, x_grouped)
    gated_sub_2 = beta.unsqueeze(-1) * (v_c - x_c)
    upd_2 = torch.einsum("bgdr,bgrp->bgdp", q, gated_sub_2)
    
    assert torch.allclose(upd_1, upd_2, atol=1e-5)
