import torch
import argparse
import json
import os
from utils.config import load_config
from utils.reproducibility import set_seed
from models.resnet import ResNetBackbone
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def run_evaluation(config, seed, checkpoint_path, output_dir):
    set_seed(seed, config["experiment"]["deterministic"])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root=config.get("dataset", {}).get("root", "./data"), train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=config["dataset"]["batch_size"], shuffle=False, num_workers=config.get("dataset", {}).get("num_workers", 2))
    
    device = torch.device(config.get("runtime", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    
    # Model
    backbone_name = config.get("model", {}).get("backbone", "ResNet8")
    hrg_after = config.get("model", {}).get("hrg_after_each_block", False)
    
    kwargs = {
        "stem_channels": config["model"]["stem_channels"],
        "stage_channels": config["model"]["stage_channels"],
        "blocks_per_stage": config["model"]["blocks_per_stage"],
        "num_classes": config["dataset"]["num_classes"],
        "hrg_after_each_block": hrg_after
    }
    
    if hrg_after and backbone_name not in ["ParameterMatchedResNet", "SENet", "CANet", "ResidualAdapterParameterMatchedResNet"]:
        hrg_kwargs = {
            "rank": config["model"]["rank"],
            "groups": config["model"]["groups"],
            "dynamic_q": config["model"]["dynamic_q"],
            "dynamic_beta": config["model"]["dynamic_beta"],
            "beta_max": config["model"]["beta_max"]
        }
        if "value_map_type" in config["model"]:
            hrg_kwargs["value_map_type"] = config["model"]["value_map_type"]
        if "q_mode" in config["model"]:
            hrg_kwargs["q_mode"] = config["model"]["q_mode"]
        kwargs["hrg_kwargs"] = hrg_kwargs
        
    if backbone_name == "VanillaResNet" or (backbone_name == "ResNet8" and not hrg_after):
        from models.resnet import ResNetBackbone
        kwargs["hrg_after_each_block"] = False
        kwargs.pop("hrg_kwargs", None)
        model = ResNetBackbone(**kwargs)
    elif backbone_name == "ParameterMatchedResNet":
        from models.baselines import ParameterMatchedResNet
        model = ParameterMatchedResNet(**kwargs)
    elif backbone_name == "ResidualAdapterParameterMatchedResNet":
        from models.baselines import ResidualAdapterParameterMatchedResNet
        if "adapter_kwargs" in config.get("model", {}):
            kwargs["adapter_kwargs"] = config["model"]["adapter_kwargs"]
        model = ResidualAdapterParameterMatchedResNet(**kwargs)
    elif backbone_name == "SENet":
        from models.baselines import SENetBackbone
        model = SENetBackbone(**kwargs)
    elif backbone_name == "CANet":
        from models.baselines import CANetBackbone
        model = CANetBackbone(**kwargs)
    else:
        from models.resnet import ResNetBackbone
        model = ResNetBackbone(**kwargs)
        
    model.to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'], strict=True)
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100.*correct/total
    
    result = {
        "seed": seed,
        "test_accuracy": acc
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=4)
        
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    run_evaluation(cfg, args.seed, args.checkpoint, args.output_dir)
