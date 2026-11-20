import torch
import numpy as np

def compute_representation_metrics(features, labels):
    """
    Computes within-class scatter (S_W), between-class scatter (S_B), and Fisher ratio.
    """
    classes = torch.unique(labels)
    C = len(classes)
    
    mu = features.mean(dim=0)
    
    s_w = 0.0
    s_b = 0.0
    
    for c in classes:
        mask = (labels == c)
        c_features = features[mask]
        n_c = c_features.size(0)
        
        if n_c == 0:
            continue
            
        mu_c = c_features.mean(dim=0)
        
        # Within-class scatter
        diff_w = c_features - mu_c
        s_w += (diff_w.norm(dim=1) ** 2).mean()
        
        # Between-class scatter
        diff_b = mu_c - mu
        s_b += (diff_b.norm() ** 2)
        
    s_w = (s_w / C).item()
    s_b = (s_b / C).item()
    fisher_ratio = s_b / (s_w + 1e-8)
    
    return {
        "s_w": s_w,
        "s_b": s_b,
        "fisher_ratio": fisher_ratio
    }

def compute_noise_sensitivity(model, x, epsilon=1e-3):
    """
    Computes ||f(x) - f(x+eps)|| / ||eps||
    """
    model.eval()
    with torch.no_grad():
        f_x = model.forward_features(x)
        
        noise = torch.randn_like(x)
        noise = noise / torch.linalg.vector_norm(noise, dim=(1,2,3), keepdim=True) * epsilon
        
        f_x_noisy = model.forward_features(x + noise)
        
        diff = (f_x - f_x_noisy).flatten(1)
        noise_flat = noise.flatten(1)
        
        sensitivity = (diff.norm(dim=1) / noise_flat.norm(dim=1)).mean().item()
        return sensitivity
