import torch
import pandas as pd
import json

class GateCollector:
    def __init__(self):
        self.epoch_stats = {}

    def collect(self, model, experiment, seed, checkpoint, split, epoch):
        layer_idx = 0
        from models.hrg_delta import HRGDeltaLayer
        for name, module in model.named_modules():
            if isinstance(module, HRGDeltaLayer):
                if module.last_beta is not None:
                    beta = module.last_beta.detach().cpu()
                    B = beta.size(0)
                    
                    key = (experiment, seed, checkpoint, split, epoch, f"layer_{layer_idx}_{name}")
                    if key not in self.epoch_stats:
                        self.epoch_stats[key] = {
                            "sum": 0.0,
                            "sq_sum": 0.0,
                            "count": 0,
                            "min": float('inf'),
                            "max": float('-inf'),
                            "hp_count": 0,
                            "orth_error_sum": 0.0,
                            "group_sum": None,
                            "rank_sum": None,
                            "num_batches": 0
                        }
                    
                    st = self.epoch_stats[key]
                    
                    st["sum"] += beta.sum().item()
                    st["sq_sum"] += (beta ** 2).sum().item()
                    st["count"] += beta.numel()
                    st["min"] = min(st["min"], beta.min().item())
                    st["max"] = max(st["max"], beta.max().item())
                    st["hp_count"] += (beta > 1.0).sum().item()
                    
                    if module.last_orthogonality_error is not None:
                        st["orth_error_sum"] += module.last_orthogonality_error * B
                    st["num_batches"] += B
                    
                    group_sum = beta.sum(dim=(0, 2))
                    if st["group_sum"] is None:
                        st["group_sum"] = group_sum
                    else:
                        st["group_sum"] += group_sum
                        
                    rank_sum = beta.sum(dim=(0, 1))
                    if st["rank_sum"] is None:
                        st["rank_sum"] = rank_sum
                    else:
                        st["rank_sum"] += rank_sum
                        
                layer_idx += 1

    def save(self, path_csv, path_json):
        if not self.epoch_stats:
            return
            
        records = []
        for key, st in self.epoch_stats.items():
            experiment, seed, checkpoint, split, epoch, layer = key
            
            mean = st["sum"] / st["count"]
            variance = (st["sq_sum"] / st["count"]) - (mean ** 2)
            std = variance ** 0.5 if variance > 0 else 0.0
            
            B_total = st["num_batches"]
            orth_mean = st["orth_error_sum"] / B_total if B_total > 0 else 0.0
            
            G = st["group_sum"].shape[0]
            R = st["rank_sum"].shape[0]
            group_means = (st["group_sum"] / (B_total * R)).tolist()
            rank_means = (st["rank_sum"] / (B_total * G)).tolist()
            
            records.append({
                "experiment": experiment,
                "seed": seed,
                "checkpoint": checkpoint,
                "split": split,
                "epoch": epoch,
                "layer": layer,
                "beta_mean": mean,
                "beta_std": std,
                "beta_min": st["min"],
                "beta_max": st["max"],
                "hyper_fraction": st["hp_count"] / st["count"],
                "orthogonality_error": orth_mean,
                "group_means": str(group_means),
                "rank_means": str(rank_means)
            })
            
        df = pd.DataFrame(records)
        df.to_csv(path_csv, index=False)
        
        net_mean = df["beta_mean"].mean()
        hp_frac = df["hyper_fraction"].mean()
        layer_means = df.groupby("layer")["beta_mean"].mean().to_dict()
        
        summary = {
            "network_mean_beta": float(net_mean),
            "hyper_projective_fraction": float(hp_frac),
            "layer_means": layer_means
        }
        with open(path_json, 'w') as f:
            json.dump(summary, f, indent=4)
