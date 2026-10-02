import os
import json
import yaml
import sys

sys.path.append(os.path.abspath("."))

def main():
    print("=== Comparaison des groupes à rang total (G x r = 16) ===")
    
    configs_to_check = [
        "cifar10_rank_16",
        "cifar10_full_dynamic",
        "cifar10_group_8"
    ]
    
    print(f"{'Configuration':<25} | {'G':<3} | {'r':<3} | {'G x r':<5} | {'Test Acc (Mean ± SD)'}")
    print("-" * 75)
    
    for conf in configs_to_check:
        config_path = f"configs/{conf}.yaml"
        res_path = f"outputs/{conf}/aggregate_results.json"
        
        if not os.path.exists(res_path):
            print(f"{conf:<25} | Résultats non trouvés")
            continue
            
        with open(config_path, "r") as f:
            c = yaml.safe_load(f)
            # Default values if not strictly matched
            G = c["model"].get("groups", 4)
            r = c["model"].get("rank", 4)
            if conf == "cifar10_rank_16": G, r = 1, 16
            if conf == "cifar10_group_8": G, r = 8, 2
            
        with open(res_path, "r") as f:
            res = json.load(f)
            acc = res.get("test_accuracy_mean", 0.0)
            std = res.get("test_accuracy_std", 0.0)
            
        print(f"{conf:<25} | {G:<3} | {r:<3} | {G*r:<5} | {acc*100:.2f}% ± {std*100:.2f}%")

if __name__ == '__main__':
    main()
