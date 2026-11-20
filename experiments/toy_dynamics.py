import os
import torch
import numpy as np
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt

def run_toy_dynamics(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    betas = [0.5, 1.0, 1.5, 2.0]
    K = 50
    dim = 10
    rank = 4
    
    records = []
    
    # Common basis
    torch.manual_seed(42)
    raw_Q = torch.randn(dim, rank)
    Q, _ = torch.linalg.qr(raw_Q, mode='reduced')
    
    q_perp_raw = torch.randn(dim, 1)
    q_perp = q_perp_raw - Q @ (Q.T @ q_perp_raw)
    q_perp = q_perp / torch.norm(q_perp)
    
    fig_active, axes_active = plt.subplots(1, 4, figsize=(16, 4))
    fig_full, axes_full = plt.subplots(1, 4, figsize=(16, 4))
    
    summary = {}
    
    for idx, beta in enumerate(betas):
        B = torch.eye(rank) * beta
        
        # Operator A = I - Q B Q^T
        A = torch.eye(dim) - Q @ B @ Q.T
        
        # --- Experiment A: Active-subspace validation ---
        c0 = torch.randn(rank)
        x0 = Q @ c0
        
        c_k_list = []
        x_k = x0.clone()
        for k in range(K):
            c_k = Q.T @ x_k
            c_k_list.append(c_k)
            x_k = A @ x_k
            
        c_k_tensor = torch.stack(c_k_list) # [K, rank]
        
        # Signed estimator
        c_k_next = c_k_tensor[1:]
        c_k_curr = c_k_tensor[:-1]
        
        lambda_hat = (c_k_next * c_k_curr).sum(dim=0) / (c_k_curr**2).sum(dim=0)
        
        assert torch.allclose(lambda_hat, torch.full_like(lambda_hat, 1.0 - beta), atol=1e-4)
        
        axes_active[idx].plot(c_k_tensor[:, 0].numpy(), marker='o', label='c_0')
        axes_active[idx].set_title(f"Beta={beta}")
        
        # --- Experiment B: Full-space decomposition ---
        c0_full = torch.randn(rank)
        c_perp_0 = torch.randn(1)
        x0_full = Q @ c0_full + q_perp @ c_perp_0
        
        c_k_full_list = []
        c_perp_k_list = []
        x_k_norm_list = []
        
        x_k_full = x0_full.clone()
        for k in range(K):
            c_k_full = Q.T @ x_k_full
            c_perp_k = q_perp.T @ x_k_full
            
            c_k_full_list.append(torch.norm(c_k_full).item())
            c_perp_k_list.append(torch.norm(c_perp_k).item())
            x_k_norm_list.append(torch.norm(x_k_full).item())
            
            x_k_full = A @ x_k_full
            
        c_perp_k_tensor = torch.tensor(c_perp_k_list)
        assert torch.allclose(c_perp_k_tensor, torch.full_like(c_perp_k_tensor, c_perp_0.abs().item()), atol=1e-4)
        
        axes_full[idx].plot(c_k_full_list, label='|c_k|')
        axes_full[idx].plot(c_perp_k_list, label='|c_perp,k|')
        axes_full[idx].plot(x_k_norm_list, label='||x_k||', linestyle='--')
        axes_full[idx].set_title(f"Beta={beta}")
        axes_full[idx].legend()
        
        summary[str(beta)] = {
            "lambda_hat_mean": lambda_hat.mean().item(),
            "expected_lambda": 1.0 - beta
        }
        
        for k in range(K):
            records.append({
                "beta": beta,
                "k": k,
                "c_k_norm": c_k_full_list[k],
                "c_perp_k_norm": c_perp_k_list[k],
                "x_k_norm": x_k_norm_list[k]
            })
            
    fig_active.savefig(os.path.join(output_dir, "active_subspace_dynamics.png"))
    fig_full.savefig(os.path.join(output_dir, "full_space_decomposition.png"))
    
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "toy_results.csv"), index=False)
    with open(os.path.join(output_dir, "toy_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    run_toy_dynamics(args.output_dir)
