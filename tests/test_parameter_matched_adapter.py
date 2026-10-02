import os
import sys
import torch
sys.path.append(os.path.abspath("."))
from models.baselines import ResidualAdapterParameterMatchedResNet
from models.hrg_delta import HRGDeltaLayer

def test_model():
    model = ResidualAdapterParameterMatchedResNet(
        stem_channels=64,
        stage_channels=[64, 128],
        blocks_per_stage=[2, 2],
        hrg_after_each_block=True,
        adapter_kwargs={
            "hidden_dims": {
                "stage1": 68,
                "stage2": 82
            }
        }
    )
    
    # Test 1: Output shape
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10), "Output shape incorrect"
    
    # Test 2 & 3: No HRG modules
    for name, module in model.named_modules():
        assert not isinstance(module, HRGDeltaLayer), f"Found HRGDeltaLayer in {name}"
        assert 'hrg' not in name.lower(), f"Found 'hrg' in module name {name}"
        assert 'qr' not in name.lower(), f"Found 'qr' in module name {name}"
        
    # Test 4: Parameter error
    target = 941194
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rel_error = abs(n_params - target) / target
    assert rel_error <= 0.01, f"Relative parameter error {rel_error} > 0.01"
    
    # Test 5 & 6: Insertion points
    insertions = 0
    for stage in model.layers:
        for module_dict in stage:
            if "attention" in module_dict:
                insertions += 1
                adapter = module_dict["attention"]
                assert adapter.conv1.in_channels == module_dict["block"].conv2.out_channels, "Channels mismatch"
    assert insertions == 4, "Number of insertions should be 4 (2 blocks per 2 stages)"
    
    # Test 7 & 8: Forward and backward
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
            assert torch.isfinite(p.grad).all(), f"Gradient not finite for {name}"
            
    print("All tests passed.")

if __name__ == "__main__":
    test_model()
