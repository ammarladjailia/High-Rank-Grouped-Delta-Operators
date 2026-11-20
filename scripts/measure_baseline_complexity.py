import os
import sys
import torch
import thop

sys.path.append(os.path.abspath("."))
from utils.config import load_config
from scripts.compute_linear_probe import get_model

def measure_complexity(model, device):
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    with torch.no_grad():
        macs, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
        
        # Warmup
        x = torch.randn(128, 3, 32, 32).to(device)
        for _ in range(100):
            _ = model(x)
        
        torch.cuda.synchronize()
        times = []
        for _ in range(1000):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            
    latency_ms = sum(times) / len(times)
    return params, macs, latency_ms

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    configs = [
        "cifar10_vanilla",
        "cifar10_senet",
        "cifar10_canet",
        "cifar10_residual_adapter_parameter_matched",
        "cifar10_full_dynamic",
        "cifar10_residual_overparameterized"
    ]
    
    import csv
    with open('complexity_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Params', 'MACs/image', 'Latency_ms'])
        
        for conf in configs:
            config_path = f"configs/{conf}.yaml"
            model = get_model(config_path)
            model = model.to(device)
            params, macs, latency = measure_complexity(model, device)
            writer.writerow([conf, params, macs, latency])
            print(f"{conf}: {params} params, {macs} MACs, {latency:.2f} ms")

if __name__ == "__main__":
    main()
