import torch
import torch.nn as nn
import torch.nn.functional as F
from .hrg_delta import HRGDeltaLayer

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetBackbone(nn.Module):
    def __init__(
        self,
        stem_channels: int,
        stage_channels: list,
        blocks_per_stage: list,
        num_classes: int = 10,
        hrg_after_each_block: bool = False,
        hrg_kwargs: dict = None,
        se_kwargs: dict = None,
        ca_kwargs: dict = None,
    ):
        super().__init__()
        self.stem_channels = stem_channels
        self.conv1 = nn.Conv2d(3, stem_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_channels)
        
        self.layers = nn.ModuleList()
        in_c = stem_channels
        
        for stage_idx, (out_c, num_blocks) in enumerate(zip(stage_channels, blocks_per_stage)):
            stage = nn.ModuleList()
            for block_idx in range(num_blocks):
                stride = 2 if block_idx == 0 and stage_idx > 0 else 1
                block = BasicBlock(in_c, out_c, stride)
                
                module_dict = nn.ModuleDict({"block": block})
                
                if hrg_after_each_block and hrg_kwargs:
                    module_dict["attention"] = HRGDeltaLayer(in_channels=out_c, **hrg_kwargs)
                elif se_kwargs:
                    from .se_net import SELayer
                    module_dict["attention"] = SELayer(out_c, **se_kwargs)
                elif ca_kwargs:
                    from .coordinate_attention import CoordAtt
                    module_dict["attention"] = CoordAtt(out_c, out_c, **ca_kwargs)
                    
                stage.append(module_dict)
                in_c = out_c
            self.layers.append(stage)
            
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, num_classes)
        self.final_channels = in_c

    def forward_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for stage in self.layers:
            for module_dict in stage:
                out = module_dict["block"](out)
                if "attention" in module_dict:
                    out = module_dict["attention"](out)
        return out

    def forward(self, x):
        features = self.forward_features(x)
        out = self.pool(features).flatten(1)
        return self.fc(out)
