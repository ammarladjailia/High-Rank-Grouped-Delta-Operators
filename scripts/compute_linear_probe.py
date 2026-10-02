import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import json
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from utils.config import load_config

def get_model(config_path):
    config = load_config(config_path)
    backbone_name = config["model"].get("backbone", "VanillaResNet")
    hrg_after = config["model"].get("hrg_after_each_block", False)
    
    kwargs = {
        "stem_channels": config["model"]["stem_channels"],
        "stage_channels": config["model"]["stage_channels"],
        "blocks_per_stage": config["model"]["blocks_per_stage"],
        "num_classes": config["dataset"]["num_classes"],
        "hrg_after_each_block": hrg_after
    }
    
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
        
    return model

def extract_features(model, dataloader, device="cpu"):
    model.eval()
    model.to(device)
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, leave=False):
            inputs = inputs.to(device)
            # Extrait les features juste avant le classifieur final
            features = model.forward_features(inputs)
            pooled = model.pool(features).flatten(1)
            all_features.append(pooled.cpu())
            all_labels.append(targets)
            
    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()

def main():
    # Force CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # We use all 50,000 for training the linear probe, 10,000 for test
    trainset = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform_val)
    testset = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform_val)
    
    trainloader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    
    configs_to_test = [
        "cifar10_vanilla",
        "cifar10_senet",
        "cifar10_canet",
        "cifar10_full_dynamic",
        "cifar10_residual_adapter_parameter_matched"
    ]
    
    # On limite l'évaluation à une seule seed (42) pour un calcul rapide
    seeds = [42]
    results = {}
    
    for conf in configs_to_test:
        results[conf] = {}
        print(f"\n=========================================")
        print(f"Évaluation de {conf}")
        print(f"=========================================")
        for seed in seeds:
            model_path = f"outputs/{conf}/seed_{seed}/checkpoint_best.pt"
            if not os.path.exists(model_path):
                print(f"Skipping seed {seed}, not found.")
                continue
                
            model = get_model(f"configs/{conf}.yaml")
            state_dict = torch.load(model_path, map_location=device)["model_state"]
            model.load_state_dict(state_dict)
            
            # Extraction
            print(f"Seed {seed} - Extraction train features...")
            X_train, y_train = extract_features(model, trainloader, device)
            print(f"Seed {seed} - Extraction test features...")
            X_test, y_test = extract_features(model, testloader, device)
            
            # Entraînement Linear Probe
            print(f"Seed {seed} - Entraînement régression logistique (L-BFGS)...")
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
            clf.fit(X_train, y_train)
            
            acc = clf.score(X_test, y_test)
            
            # Calcul des métriques de représentation (Fisher, Marges)
            from metrics.representation import compute_representation_metrics
            rep_metrics = compute_representation_metrics(torch.tensor(X_test), torch.tensor(y_test))
            
            results[conf][seed] = {
                "linear_probe_acc": acc,
                "fisher_ratio": rep_metrics["fisher_ratio"],
                "s_w": rep_metrics["s_w"],
                "s_b": rep_metrics["s_b"]
            }
            
            print(f"Seed {seed} - Linear Probe Acc: {acc*100:.2f}% | Fisher: {rep_metrics['fisher_ratio']:.4f} | S_w: {rep_metrics['s_w']:.4f} | S_b: {rep_metrics['s_b']:.4f}")
            
    with open("outputs/representation_and_probe_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n\n--- SUMMARY ---")
    for conf in results:
        seed_results = list(results[conf].values())
        if len(seed_results) > 0:
            mean_acc = sum(r["linear_probe_acc"] for r in seed_results) / len(seed_results)
            mean_fisher = sum(r["fisher_ratio"] for r in seed_results) / len(seed_results)
            print(f"{conf}: Mean Linear Probe Acc = {mean_acc*100:.2f}% | Mean Fisher = {mean_fisher:.4f}")

if __name__ == "__main__":
    main()
