import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNetBackbone

class ParameterMatchedResNet(ResNetBackbone):
    """
    Adjusts width or adds a residual module to match HRG-Delta parameter count.
    Here we add a standard residual module after each block where HRG would be.
    """
    def __init__(self, *args, **kwargs):
        # We don't use hrg_after_each_block here, but we mimic the insertion points
        hrg_after = kwargs.pop("hrg_after_each_block", False)
        super().__init__(*args, **kwargs)
        
        if hrg_after:
            for stage in self.layers:
                for module_dict in stage:
                    out_c = module_dict["block"].conv2.out_channels
                    # Adding a generic module (e.g., lightweight conv block) to match params
                    # Let's add a basic block to parameter match
                    from .resnet import BasicBlock
                    module_dict["attention"] = BasicBlock(out_c, out_c)

from .se_net import SELayer
from .coordinate_attention import CoordAtt

class SENetBackbone(ResNetBackbone):
    def __init__(self, *args, **kwargs):
        hrg_after = kwargs.pop("hrg_after_each_block", False)
        super().__init__(*args, **kwargs)
        if hrg_after:
            for stage in self.layers:
                for module_dict in stage:
                    out_c = module_dict["block"].conv2.out_channels
                    module_dict["attention"] = SELayer(out_c)

class CANetBackbone(ResNetBackbone):
    def __init__(self, *args, **kwargs):
        hrg_after = kwargs.pop("hrg_after_each_block", False)
        super().__init__(*args, **kwargs)
        if hrg_after:
            for stage in self.layers:
                for module_dict in stage:
                    out_c = module_dict["block"].conv2.out_channels
                    module_dict["attention"] = CoordAtt(out_c, out_c)


class ResidualBottleneckAdapter(nn.Module):
    def __init__(self, channels, hidden_channels, residual_scale=1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.residual_scale = residual_scale
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        return x + self.residual_scale * out

class ResidualAdapterParameterMatchedResNet(ResNetBackbone):
    def __init__(self, *args, **kwargs):
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        hrg_after = kwargs.pop("hrg_after_each_block", False)
        super().__init__(*args, **kwargs)
        
        hidden_dims = adapter_kwargs.get("hidden_dims", {})
        residual_scale = adapter_kwargs.get("residual_scale", 1.0)
        
        if hrg_after:
            for stage_idx, stage in enumerate(self.layers):
                stage_key = f"stage{stage_idx + 1}"
                hidden_channels = hidden_dims.get(stage_key, 16)
                for module_dict in stage:
                    out_c = module_dict["block"].conv2.out_channels
                    module_dict["attention"] = ResidualBottleneckAdapter(out_c, hidden_channels, residual_scale=residual_scale)
