import os
import sys
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from scripts.compute_linear_probe import get_model # Re-use model loader

def get_betas_from_model(model, dataloader, device):
    """
    Passe les données dans le modèle et récupère toutes les valeurs de beta (portes)
    """
    model.eval()
    model.to(device)
    
    all_betas = []
    
    # On ajoute un hook (espion) sur toutes les couches HRGDeltaLayer
    hooks = []
    def hook_fn(module, input, output):
        if hasattr(module, 'last_beta') and module.last_beta is not None:
            all_betas.append(module.last_beta.detach().cpu().flatten())
            
    from models.hrg_delta import HRGDeltaLayer
    for name, module in model.named_modules():
        if isinstance(module, HRGDeltaLayer):
            hooks.append(module.register_forward_hook(hook_fn))
            
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader, desc="Inference"):
            inputs = inputs.to(device)
            _ = model(inputs)
            
    for h in hooks:
        h.remove()
        
    if len(all_betas) > 0:
        return torch.cat(all_betas).numpy()
    return None

def main():
    print("=== Examen des distributions effectives de Beta ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Un petit subset de test suffit (ex: 1000 images)
    testset = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform_val)
    subset = torch.utils.data.Subset(testset, range(1000))
    testloader = DataLoader(subset, batch_size=256, shuffle=False)
    
    configs = ["cifar10_beta_01", "cifar10_beta_02", "cifar10_beta_03", "cifar10_full_dynamic"]
    seed = 42
    
    plt.figure(figsize=(10, 6))
    
    for conf in configs:
        model_path = f"outputs/{conf}/seed_{seed}/checkpoint_best.pt"
        if not os.path.exists(model_path):
            print(f"Skipping {conf} - model not found")
            continue
            
        print(f"Extraction des betas pour {conf}...")
        model = get_model(f"configs/{conf}.yaml")
        
        # Load weights
        state_dict = torch.load(model_path, map_location=device)["model_state"]
        model.load_state_dict(state_dict)
        
        betas = get_betas_from_model(model, testloader, device)
        if betas is not None:
            plt.hist(betas, bins=100, alpha=0.5, label=f"{conf} (mean={betas.mean():.2f})", density=True)
            
    plt.title(r"Effective Gate ($\beta$) Distribution at Inference")
    plt.xlabel(r"Gate Value ($\beta$)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_paths = [
        "outputs/beta_distributions.png",
        "Revision/Tex_Revision/figures/beta_distributions.png",
        "Review_Color/figures/beta_distributions.png",
        "tmp_unzip/Tex_Revision/figures/beta_distributions.png",
    ]
    for out_img in out_paths:
        if os.path.exists(os.path.dirname(out_img)):
            plt.savefig(out_img, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {out_img}")

if __name__ == '__main__':
    main()
