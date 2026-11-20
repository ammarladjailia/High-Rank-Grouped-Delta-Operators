import torch
import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from utils.config import load_config
from utils.reproducibility import set_seed
from models.resnet import ResNetBackbone
from models.hrg_delta import HRGDeltaLayer
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from metrics.gating import GateCollector
from metrics.spectral import active_spectral_proxy
from metrics.representation import compute_representation_metrics, compute_noise_sensitivity

def run_evaluation(config, seed, checkpoint_path, output_dir):
    set_seed(seed, config["experiment"]["deterministic"])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(root=config.get("dataset", {}).get("root", "./data"), train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=config["dataset"]["batch_size"], shuffle=False, num_workers=config["dataset"].get("num_workers", 2))
    
    device = torch.device(config["runtime"]["device"])
    
    backbone_name = config.get("model", {}).get("backbone", "ResNet8")
    hrg_after = config.get("model", {}).get("hrg_after_each_block", False)
    
    kwargs = {
        "stem_channels": config["model"]["stem_channels"],
        "stage_channels": config["model"]["stage_channels"],
        "blocks_per_stage": config["model"]["blocks_per_stage"],
        "num_classes": config["dataset"]["num_classes"],
        "hrg_after_each_block": hrg_after
    }
    
    if "adapter_kwargs" in config["model"]:
        kwargs["adapter_kwargs"] = config["model"]["adapter_kwargs"]
    
    if hrg_after and backbone_name not in ["ParameterMatchedResNet", "SENet", "CANet", "ResidualAdapterParameterMatchedResNet"]:
        kwargs["hrg_kwargs"] = {
            "rank": config["model"]["rank"],
            "groups": config["model"]["groups"],
            "dynamic_q": config["model"]["dynamic_q"],
            "dynamic_beta": config["model"]["dynamic_beta"],
            "beta_max": config["model"]["beta_max"]
        }
        
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
    
    gate_collector = GateCollector()
    
    correct = 0
    total = 0
    
    all_features = []
    all_labels = []
    layer_max_proxies = {}
    
    # 1 batch of noise sensitivity
    noise_sensitivities = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Gating collection
            gate_collector.collect(model, config["experiment"]["name"], seed, checkpoint_path, "test", epoch=100)
            
            # Spectral proxies
            layer_idx = 0
            for name, module in model.named_modules():
                if isinstance(module, HRGDeltaLayer) and module.last_beta is not None:
                    proxy = active_spectral_proxy(module.last_beta).max().item()
                    l_key = f"layer_{layer_idx}_{name}"
                    layer_max_proxies[l_key] = max(layer_max_proxies.get(l_key, 0.0), proxy)
                    layer_idx += 1
            
            # Features
            if hasattr(model, 'forward_features'):
                f_x = model.forward_features(inputs)
                all_features.append(f_x.cpu())
                all_labels.append(targets.cpu())
                
                # compute noise sensitivity for the batch
                if i < 10:  # Compute over first 10 batches to save time
                    ns = compute_noise_sensitivity(model, inputs)
                    noise_sensitivities.append(ns)
            
    acc = 100. * correct / total
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. gating statistics
    gate_collector.save(
        os.path.join(output_dir, "gating_statistics.csv"),
        os.path.join(output_dir, "gating_summary.json")
    )
    
    # 2. test_result.json
    with open(os.path.join(output_dir, "test_result.json"), 'w') as f:
        json.dump({"seed": seed, "test_accuracy": acc}, f, indent=4)
        
    # 3. spectral_proxy.json
    max_proxy = max(layer_max_proxies.values()) if layer_max_proxies else 0.0
    with open(os.path.join(output_dir, "spectral_proxy.json"), 'w') as f:
        json.dump({
            "active_fixed_basis_spectral_proxy": max_proxy,
            "layer_proxies": layer_max_proxies
        }, f, indent=4)
        
    # 4. representation_metrics.json
    if all_features:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        rep_metrics = compute_representation_metrics(all_features, all_labels)
        if noise_sensitivities:
            rep_metrics["noise_sensitivity"] = sum(noise_sensitivities) / len(noise_sensitivities)
            
        with open(os.path.join(output_dir, "representation_metrics.json"), 'w') as f:
            json.dump(rep_metrics, f, indent=4)
            
    # 5. evaluation_metadata.json
    import subprocess
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode('utf-8').strip()
    except:
        gpu_info = "Unknown"
        
    metadata = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_info,
        "config": config,
        "seed": seed
    }
    with open(os.path.join(output_dir, "evaluation_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    run_evaluation(cfg, args.seed, args.checkpoint, args.output_dir)
