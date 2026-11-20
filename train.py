import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import json

from utils.config import load_config
from utils.reproducibility import set_seed
from utils.environment import save_environment_metadata
from utils.checkpointing import save_checkpoint
from utils.divergence import DivergenceMonitor
from utils.logging import get_logger
from models.resnet import ResNetBackbone
from metrics.gating import GateCollector
from metrics.complexity import measure_latency, count_parameters
from metrics.spectral import active_spectral_proxy
import evaluate

logger = get_logger("train")

from torch.utils.data import Dataset

class CustomSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        super().__init__()
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, target

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    
    if 'optimizer_state' in checkpoint and optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    elif 'optimizer' in checkpoint and optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    if 'scheduler_state' in checkpoint and scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    elif 'scheduler' in checkpoint and scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])
        
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        
    start_epoch = checkpoint.get('epoch', -1) + 1
    best_val_accuracy = checkpoint.get(
        'best_val_accuracy',
        checkpoint.get('val_accuracy', checkpoint.get('best_val_acc', -float('inf')))
    )
    if best_val_accuracy is None:
        best_val_accuracy = -float('inf')
        
    return start_epoch, best_val_accuracy

def run_one_seed(config, seed, args):
    exp_name = config["experiment"]["name"]
    output_dir = Path(config["experiment"]["output_dir"]) / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config_resolved.yaml", "w") as f:
        import yaml
        yaml.dump(config, f)
        
    save_environment_metadata(output_dir / "environment_metadata.json")
    set_seed(seed, config["experiment"]["deterministic"])
    
    device = torch.device(config["runtime"]["device"])
    use_amp = config["runtime"]["use_amp"] and device.type == "cuda"
    
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if args.dry_run:
        full_trainset = torchvision.datasets.FakeData(size=1000, image_size=(3, 32, 32), num_classes=10, transform=None)
    else:
        full_trainset = torchvision.datasets.CIFAR10(root=config["dataset"]["root"], train=True, download=True, transform=None)
    
    # Split
    val_frac = config["dataset"]["validation_fraction"]
    num_train = len(full_trainset)
    indices = list(range(num_train))
    split = int(num_train * (1 - val_frac))
    
    # Generator for split
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_train, generator=gen).tolist()
    
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_subset = CustomSubset(full_trainset, train_idx, transform_train)
    val_subset = CustomSubset(full_trainset, val_idx, transform_val)
    
    nw = 0 if args.dry_run else config["dataset"]["num_workers"]
    trainloader = DataLoader(train_subset, batch_size=config["dataset"]["batch_size"], shuffle=True, num_workers=nw, pin_memory=config["dataset"]["pin_memory"])
    valloader = DataLoader(val_subset, batch_size=config["dataset"]["batch_size"], shuffle=False, num_workers=nw, pin_memory=config["dataset"]["pin_memory"])
    
    # Model
    backbone_name = config["model"].get("backbone", "ResNet8")
    hrg_after = config.get("model", {}).get("hrg_after_each_block", False)
    
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
    model.to(device)
    
    total_params = count_parameters(model)
    logger.info(f"Total trainable parameters: {total_params}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["training"]["lr"], momentum=config["training"]["momentum"], weight_decay=config["training"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    
    divergence_monitor = DivergenceMonitor(
        loss_threshold=config["stability"]["divergence_loss_threshold"],
        patience=config["stability"]["divergence_patience_epochs"]
    )
    
    gate_collector = GateCollector()
    
    start_epoch = 0
    best_val_acc = -float('inf')
    
    if args.resume is not None:
        start_epoch, best_val_acc = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device
        )
        
    for epoch in range(start_epoch, config["training"]["epochs"]):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if args.dry_run and args.max_train_batches and batch_idx >= args.max_train_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda", enabled=use_amp) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            if scaler:
                scaler.scale(loss).backward()
                if config["training"]["gradient_clip_norm"] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config["training"]["gradient_clip_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"])
                optimizer.step()
                
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        avg_train_loss = train_loss / (batch_idx + 1)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        max_proxy = 0.0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                if args.dry_run and args.max_val_batches and batch_idx >= args.max_val_batches:
                    break
                    
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                gate_collector.collect(model, exp_name, seed, "checkpoint", "val", epoch)
                
                # compute max active proxy
                from models.hrg_delta import HRGDeltaLayer
                for mod_name, module in model.named_modules():
                    if isinstance(module, HRGDeltaLayer) and module.last_beta is not None:
                        proxy = active_spectral_proxy(module.last_beta).max().item()
                        max_proxy = max(max_proxy, proxy)
                        
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / (batch_idx + 1) if val_total > 0 else float('inf')
        
        scheduler.step()
        
        logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.2f}% | Max Proxy: {max_proxy:.4f}")
        save_checkpoint(model, optimizer, scheduler, seed, epoch, config, val_acc, output_dir / "checkpoint_last.pt")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, scheduler, seed, epoch, config, val_acc, output_dir / "checkpoint_best.pt")
            
        if divergence_monitor.check(epoch, avg_train_loss, max_proxy):
            logger.warning(f"Divergence detected: {divergence_monitor.divergence_reason}")
            break
            
        if args.dry_run:
            logger.info("Dry run complete.")
            # measure latency as part of dry-run if configured
            if config["evaluation"]["measure_latency"]:
                lat = measure_latency(model, torch.randn(1, 3, 32, 32).to(device))
                logger.info(f"Latency (B=1): {lat:.2f} ms")
            break
            
    gate_collector.save(output_dir / "gating_statistics.csv", output_dir / "gating_summary.json")
    
    if not args.dry_run:
        # evaluate on test
        evaluate.run_evaluation(config, seed, output_dir / "checkpoint_best.pt", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Optional subset of seeds to execute.")
    args = parser.parse_args()
    
    config = load_config(args.config)
    seeds = args.seeds if args.seeds else config["experiment"]["seeds"]
    for seed in seeds:
        run_one_seed(config, seed, args)
        if args.dry_run:
            break # only one seed for dry run
