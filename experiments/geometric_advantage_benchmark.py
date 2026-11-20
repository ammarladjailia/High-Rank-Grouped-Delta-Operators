import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def generate_spiked_subspace_dataset(num_samples=5000, G=4, dg=16, r=4, num_classes=10, noise_scale=3.0, seed=42):
    set_seed(seed)
    C = G * dg
    prototypes = F.normalize(torch.randn(num_classes, G, r), dim=-1)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # 8-dim orientation context that controls the rotating subspace
    Z = torch.randn(num_samples, 8)
    W_orient = torch.randn(8, G * dg * r)
    
    X_feats = torch.zeros(num_samples, G, dg)
    
    for i in range(num_samples):
        y = labels[i]
        z_i = Z[i]
        raw_U = (z_i @ W_orient).view(G, dg, r)
        Q_true, _ = torch.linalg.qr(raw_U)
        
        for g in range(G):
            # Informative rank-r signal in active subspace
            sig = Q_true[g] @ (prototypes[y, g] + 0.05 * torch.randn(r))
            
            # High-energy distractor strictly in orthogonal complement Q_perp
            dist_raw = torch.randn(dg)
            dist_perp = dist_raw - Q_true[g] @ (Q_true[g].T @ dist_raw)
            distractor = F.normalize(dist_perp, dim=0) * (noise_scale * torch.randn(1).abs() + 1.0)
            
            X_feats[i, g] = sig + distractor
            
    X_full = torch.cat([Z, X_feats.view(num_samples, -1)], dim=-1) # [N, 8 + 64] = [N, 72]
    return X_full, labels

# 1. Vanilla ResNet/MLP Backbone
class VanillaModel(nn.Module):
    def __init__(self, in_dim=72, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# 2. Magnitude-Only Channel Attention (SE-Net)
class SENetModel(nn.Module):
    def __init__(self, in_dim=72, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.se = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        # Magnitude scaling across channels in fixed Cartesian basis
        w = self.se(h)
        h = h * w
        h = F.relu(self.bn2(self.fc2(h)))
        return self.classifier(h)

# 3. Coordinate Attention (CA-Net)
class CANetModel(nn.Module):
    def __init__(self, in_dim=72, hidden_dim=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.ca_pool = nn.Linear(hidden_dim, hidden_dim // 4)
        self.ca_act = nn.ReLU()
        self.ca_out = nn.Linear(hidden_dim // 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        attn = torch.sigmoid(self.ca_out(self.ca_act(self.ca_pool(h))))
        h = h * attn
        h = F.relu(self.bn2(self.fc2(h)))
        return self.classifier(h)

# 4. Deep Delta Learning (DDL): Rank 1, 1 Group (r=1, G=1)
class DDLModel(nn.Module):
    def __init__(self, in_dim=72, ctx_dim=8, feat_dim=64, num_classes=10):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.feat_dim = feat_dim
        self.q_gen = nn.Linear(ctx_dim, feat_dim)
        self.beta_gen = nn.Linear(ctx_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(ctx_dim + feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        z = x[:, :self.ctx_dim]
        feat = x[:, self.ctx_dim:]
        
        q = F.normalize(self.q_gen(z), dim=-1) # [B, feat_dim]
        beta = 2.0 * torch.sigmoid(self.beta_gen(z)) # [B, 1]
        
        proj = (feat * q).sum(dim=-1, keepdim=True) # [B, 1]
        update = beta * proj * q # Rank-1 subspace update
        
        filtered_feat = feat + update
        return self.fc(torch.cat([z, filtered_feat], dim=-1))

# 5. HRG-Delta: Rank 4, 4 Groups (r=4, G=4, Ours)
class HRGDeltaModel(nn.Module):
    def __init__(self, in_dim=72, ctx_dim=8, G=4, dg=16, r=4, num_classes=10):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.G = G
        self.dg = dg
        self.r = r
        
        self.q_gen = nn.Linear(ctx_dim, G * dg * r)
        self.beta_gen = nn.Linear(ctx_dim, G * r)
        self.value_gen = nn.Linear(G * dg, G * dg)
        
        self.fc = nn.Sequential(
            nn.Linear(G * r, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        b = x.size(0)
        z = x[:, :self.ctx_dim]
        feat = x[:, self.ctx_dim:].view(b, self.G, self.dg)
        
        raw_q = self.q_gen(z).view(b, self.G, self.dg, self.r)
        Q, _ = torch.linalg.qr(raw_q) # [B, G, dg, r] orthonormal basis
        Beta = 2.0 * torch.sigmoid(self.beta_gen(z)).view(b, self.G, self.r) # [B, G, r]
        
        # Subspace projection and gating
        proj = torch.einsum("bgdr,bgd->bgr", Q, feat) # [B, G, r]
        gated_subspace = Beta * proj # [B, G, r]
        
        return self.fc(gated_subspace.reshape(b, -1))

def train_and_eval(model, train_x, train_y, test_x, test_y, device, epochs=30):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    n_train = train_x.size(0)
    batch_size = 64
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = train_x[idx].to(device), train_y[idx].to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_xb = test_x.to(device)
        test_yb = test_y.to(device)
        preds = model(test_xb).argmax(dim=-1)
        acc = (preds == test_yb).float().mean().item() * 100.0
    return acc

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Structured Multi-Subspace Spiked Covariance Benchmark (Device: {device}) ===")
    
    seeds = [42, 123, 456, 789, 2026]
    models = {
        "Vanilla ResNet": lambda: VanillaModel(),
        "SE-Net (Magnitude-only)": lambda: SENetModel(),
        "CA-Net (Coordinate-Attn)": lambda: CANetModel(),
        "DDL (Rank-1, G=1)": lambda: DDLModel(),
        "HRG-Delta (Rank-4, G=4, Ours)": lambda: HRGDeltaModel()
    }
    
    results = {m: [] for m in models}
    
    for s in seeds:
        print(f"\n--- Running Seed {s} ---")
        train_x, train_y = generate_spiked_subspace_dataset(num_samples=4000, seed=s)
        test_x, test_y = generate_spiked_subspace_dataset(num_samples=1000, seed=s + 10000)
        
        for name, constructor in models.items():
            set_seed(s)
            model = constructor()
            acc = train_and_eval(model, train_x, train_y, test_x, test_y, device, epochs=30)
            results[name].append(acc)
            print(f"  {name:<30}: {acc:.2f}%")
            
    records = []
    print("\n" + "="*80)
    print("FINAL 5-SEED RESULTS FOR GEOMETRIC ADVANTAGE BENCHMARK")
    print("="*80)
    print(f"{'Model':<30} | {'Test Accuracy (Mean ± SD)':<25} | {'All 5 Seeds'}")
    print("-" * 80)
    for name, accs in results.items():
        mean_acc = np.mean(accs)
        std_acc = np.std(accs, ddof=1)
        seed_str = ", ".join([f"{a:.1f}%" for a in accs])
        print(f"{name:<30} | {mean_acc:.2f}% ± {std_acc:.2f}%{'':<10} | [{seed_str}]")
        records.append({
            "Model": name,
            "Accuracy_Mean": mean_acc,
            "Accuracy_SD": std_acc,
            "Seeds": accs
        })
        
    out_dir = "outputs/geometric_benchmark"
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "geometric_advantage_5seeds.csv"), index=False)
    with open(os.path.join(out_dir, "geometric_advantage_summary.json"), "w") as f:
        json.dump(records, f, indent=4)
    print(f"\nResults successfully saved to {out_dir}/geometric_advantage_5seeds.csv")

if __name__ == "__main__":
    run_benchmark()
