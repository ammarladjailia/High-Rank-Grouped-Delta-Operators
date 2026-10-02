import sys
import os
import torch
import json
import subprocess

def save_environment_metadata(path_json: str):
    metadata = {
        "hostname": os.uname().nodename,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "venv_path": os.environ.get("VIRTUAL_ENV"),
        "is_virtual_environment": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        metadata["gpu_total_memory_gib"] = round(props.total_memory / 1024**3, 2)
        metadata["compute_capability"] = f"{props.major}.{props.minor}"
        metadata["cudnn_version"] = torch.backends.cudnn.version()
        metadata["deterministic_algorithm"] = torch.are_deterministic_algorithms_enabled()
        
        try:
            nvidia_smi = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
            metadata["nvidia_smi_output"] = nvidia_smi
        except Exception as e:
            metadata["nvidia_smi_output"] = str(e)
            
    with open(path_json, 'w') as f:
        json.dump(metadata, f, indent=4)
