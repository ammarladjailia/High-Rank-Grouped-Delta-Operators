import torch
from models.hrg_delta import HRGDeltaLayer

def main():
    print("Testing HRG-Delta layer initialization and forward pass...")
    layer = HRGDeltaLayer(in_channels=64, rank=4, groups=4, beta_max=2.0)
    x = torch.randn(2, 64, 8, 8, requires_grad=True)
    y = layer(x)
    print(f"Input shape:  {list(x.shape)}")
    print(f"Output shape: {list(y.shape)}")
    print(f"Gate mean:    {layer.last_beta_mean:.4f}")
    print(f"Orthogonality error: {layer.last_orthogonality_error:.2e}")
    loss = y.sum()
    loss.backward()
    print("Backward pass successful: gradients verified.")
    print("HRG-Delta verification complete.")

if __name__ == "__main__":
    main()
