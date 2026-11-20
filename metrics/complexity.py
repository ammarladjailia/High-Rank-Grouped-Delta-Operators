import time
import torch

@torch.no_grad()
def measure_latency(model, example_input, warmup=100, iterations=1000):
    model.eval()
    device = example_input.device

    for _ in range(warmup):
        _ = model(example_input)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    for _ in range(iterations):
        _ = model(example_input)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return 1000.0 * (time.perf_counter() - start) / iterations

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

