import os
import sys
sys.path.append(os.path.abspath("."))
import torch

from models.resnet import ResNetBackbone
from models.baselines import ResidualAdapterParameterMatchedResNet

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    target = 941194
    print(f"HRG parameters: {target}")
    
    vanilla_model = ResNetBackbone(
        stem_channels=64,
        stage_channels=[64, 128],
        blocks_per_stage=[2, 2],
        hrg_after_each_block=False
    )
    print(f"Vanilla parameters: {count_trainable_parameters(vanilla_model)}")
    
    stage1_candidates = list(range(4, 128, 2))
    stage2_candidates = list(range(4, 128, 2))
    
    best = None
    
    print("candidate hidden dimensions (stage1, stage2) | adapter parameters | absolute difference | relative difference (%)")
    
    for stage1_width in stage1_candidates:
        for stage2_width in stage2_candidates:
            model = ResidualAdapterParameterMatchedResNet(
                stem_channels=64,
                stage_channels=[64, 128],
                blocks_per_stage=[2, 2],
                hrg_after_each_block=True,
                adapter_kwargs={
                    "hidden_dims": {
                        "stage1": stage1_width,
                        "stage2": stage2_width
                    }
                }
            )
            n_params = count_trainable_parameters(model)
            diff = abs(n_params - target)
            relative_error = diff / target
            
            candidate = {
                "stage1_width": stage1_width,
                "stage2_width": stage2_width,
                "params": n_params,
                "diff": diff,
                "relative_error": relative_error,
            }
            
            if relative_error <= 0.01:
                # print(f"({stage1_width}, {stage2_width}) | {n_params} | {diff} | {relative_error*100:.3f}%")
                pass
                
            if best is None or candidate["relative_error"] < best["relative_error"]:
                best = candidate
                
    print("\nBest candidate:")
    print(best)

if __name__ == "__main__":
    main()
